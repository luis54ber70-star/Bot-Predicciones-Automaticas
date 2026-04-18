#!/usr/bin/env python3
"""
Bot de Predicciones Deportivas
==============================
Sistema automatizado que obtiene cuotas de The Odds API, filtra predicciones
seguras basadas en probabilidad implícita, consulta resultados reales con
All Sport API, y envía notificaciones por Telegram.

Modos de ejecución:
    python bot.py futbol       -> Predicciones de fútbol (10:00 AM CDMX)
    python bot.py mlb_nba      -> Predicciones MLB y NBA (02:00 AM CDMX)
    python bot.py aprendizaje  -> Análisis de resultados (11:00 PM CDMX)

Variables de entorno requeridas:
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ODDS_API_KEY, ALL_SPORT_API_KEY, GH_PAT
"""

import os
import sys
import csv
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ALL_SPORT_API_KEY = os.environ.get("ALL_SPORT_API_KEY", "")
GH_PAT = os.environ.get("GH_PAT", "")

# ---------------------------------------------------------------------------
# Constantes de API
# ---------------------------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ALLSPORT_API_BASE = "https://apiv2.allsportsapi.com"

# Umbral mínimo de probabilidad implícita para considerar una predicción "segura"
SAFE_PROBABILITY_THRESHOLD = 0.60

# Mapeo de sport_key (The Odds API) -> deporte (All Sport API)
SPORT_KEY_TO_ALLSPORT = {
    "soccer_epl": "football",
    "soccer_spain_la_liga": "football",
    "soccer_germany_bundesliga": "football",
    "soccer_italy_serie_a": "football",
    "soccer_france_ligue_one": "football",
    "soccer_mexico_ligamx": "football",
    "soccer_uefa_champs_league": "football",
    "baseball_mlb": "baseball",
    "basketball_nba": "basketball",
}

# Ligas de fútbol a consultar
FOOTBALL_SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_mexico_ligamx",
    "soccer_uefa_champs_league",
]

HISTORIAL_FILE = "historial.csv"
ANALYSIS_FILE = "analisis.csv"

HISTORIAL_COLUMNS = [
    "event_id",
    "sport_key",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "team_predicted",
    "odds",
    "implied_probability",
    "prediction_date",
    "actual_winner",
    "prediction_correct",
    "analysis_date",
]


# ===================================================================
# MÓDULO 1: Telegram
# ===================================================================

def send_telegram(message: str) -> None:
    """Envía un mensaje al chat de Telegram configurado."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado. Mensaje no enviado.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    # Telegram limita mensajes a 4096 caracteres
    chunks = [message[i:i + 4000] for i in range(0, len(message), 4000)]
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "Markdown",
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            logger.info("Mensaje de Telegram enviado correctamente.")
        except requests.RequestException as exc:
            logger.error("Error enviando Telegram: %s", exc)


# ===================================================================
# MÓDULO 2: The Odds API — Obtención de cuotas
# ===================================================================

def fetch_odds(sport_key: str) -> list:
    """Obtiene las cuotas head-to-head de The Odds API para un deporte."""
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY no configurada.")
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us,eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.info(
            "Cuotas obtenidas para %s: %d eventos (requests restantes: %s)",
            sport_key, len(events), remaining,
        )
        return events
    except requests.RequestException as exc:
        logger.error("Error obteniendo cuotas para %s: %s", sport_key, exc)
        return []


# ===================================================================
# MÓDULO 3: Cálculo de probabilidad implícita y filtrado
# ===================================================================

def implied_probability(decimal_odds: float) -> float:
    """Calcula la probabilidad implícita a partir de cuotas decimales."""
    if decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


def get_historical_accuracy(sport_key: str, team: str) -> float:
    """
    Consulta el historial para obtener la precisión histórica de predicciones
    para un equipo y deporte específico. Devuelve un factor de ajuste.
    """
    if not os.path.exists(HISTORIAL_FILE):
        return 1.0  # Sin historial, no ajustar

    try:
        df = pd.read_csv(HISTORIAL_FILE)
        team_history = df[
            (df["sport_key"] == sport_key)
            & (df["team_predicted"] == team)
            & (df["prediction_correct"].notna())
        ]
        if len(team_history) < 3:
            return 1.0  # Datos insuficientes

        accuracy = team_history["prediction_correct"].astype(float).mean()
        # Factor de ajuste: si el equipo acierta mucho, subir; si falla, bajar
        return 0.7 + (accuracy * 0.6)  # Rango: 0.7 (0% accuracy) a 1.3 (100%)
    except Exception:
        return 1.0


def filter_safe_predictions(events: list) -> list:
    """
    Filtra predicciones seguras: probabilidad implícita > umbral,
    ajustada por datos históricos del equipo.
    Solo toma la mejor predicción por evento (evita duplicados).
    """
    safe = []
    seen_events = set()

    for event in events:
        event_id = event.get("id", "")
        sport_key = event.get("sport_key", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        best_for_event = None
        best_prob = 0.0

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = outcome.get("name", "")
                    odds = outcome.get("price", 0)
                    if odds <= 1.0:
                        continue

                    prob = implied_probability(odds)
                    hist_factor = get_historical_accuracy(sport_key, team)
                    adjusted_prob = prob * hist_factor

                    if adjusted_prob > SAFE_PROBABILITY_THRESHOLD and adjusted_prob > best_prob:
                        best_prob = adjusted_prob
                        best_for_event = {
                            "event_id": event_id,
                            "sport_key": sport_key,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": bookmaker.get("title", ""),
                            "team_predicted": team,
                            "odds": odds,
                            "implied_probability": round(prob, 4),
                            "prediction_date": datetime.now(timezone.utc).isoformat(),
                            "actual_winner": "",
                            "prediction_correct": "",
                            "analysis_date": "",
                        }

        if best_for_event and event_id not in seen_events:
            safe.append(best_for_event)
            seen_events.add(event_id)

    logger.info("Predicciones seguras encontradas: %d", len(safe))
    return safe


# ===================================================================
# MÓDULO 4: All Sport API — Resultados reales
# ===================================================================

def fetch_results(allsport_sport: str, date_from: str, date_to: str) -> list:
    """Obtiene resultados de partidos de All Sport API."""
    if not ALL_SPORT_API_KEY:
        logger.error("ALL_SPORT_API_KEY no configurada.")
        return []

    url = f"{ALLSPORT_API_BASE}/{allsport_sport}/"
    params = {
        "met": "Fixtures",
        "APIkey": ALL_SPORT_API_KEY,
        "from": date_from,
        "to": date_to,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", [])
        if results is None:
            results = []
        logger.info(
            "Resultados obtenidos para %s (%s a %s): %d partidos",
            allsport_sport, date_from, date_to, len(results),
        )
        return results
    except requests.RequestException as exc:
        logger.error("Error obteniendo resultados de All Sport API: %s", exc)
        return []


def determine_winner(result_entry: dict) -> str:
    """Determina el ganador de un partido a partir de los datos de All Sport API."""
    final = result_entry.get("event_final_result", "")
    if not final or " - " not in final:
        return ""

    try:
        parts = final.split(" - ")
        home_score = int(parts[0].strip())
        away_score = int(parts[1].strip())
    except (ValueError, IndexError):
        return ""

    if home_score > away_score:
        return result_entry.get("event_home_team", "")
    elif away_score > home_score:
        return result_entry.get("event_away_team", "")
    else:
        return "Empate"


# ===================================================================
# MÓDULO 5: Gestión del historial CSV
# ===================================================================

def load_historial() -> pd.DataFrame:
    """Carga el historial de predicciones desde CSV."""
    if os.path.exists(HISTORIAL_FILE):
        try:
            df = pd.read_csv(HISTORIAL_FILE)
            # Asegurar que todas las columnas existan
            for col in HISTORIAL_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception as exc:
            logger.error("Error leyendo %s: %s", HISTORIAL_FILE, exc)

    return pd.DataFrame(columns=HISTORIAL_COLUMNS)


def save_predictions(predictions: list) -> None:
    """Agrega nuevas predicciones al historial CSV evitando duplicados."""
    df_existing = load_historial()
    df_new = pd.DataFrame(predictions)

    # Asegurar columnas
    for col in HISTORIAL_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ""

    # Evitar duplicados por event_id + team_predicted
    if not df_existing.empty:
        existing_keys = set(
            zip(df_existing["event_id"], df_existing["team_predicted"])
        )
        df_new = df_new[
            ~df_new.apply(
                lambda r: (r["event_id"], r["team_predicted"]) in existing_keys,
                axis=1,
            )
        ]

    if df_new.empty:
        logger.info("No hay predicciones nuevas para agregar.")
        return

    df_combined = pd.concat(
        [df_existing, df_new[HISTORIAL_COLUMNS]], ignore_index=True
    )
    df_combined.to_csv(HISTORIAL_FILE, index=False)
    logger.info(
        "Historial actualizado: %d nuevas, %d total.",
        len(df_new), len(df_combined),
    )


# ===================================================================
# MÓDULO 6: Aprendizaje — Comparación de predicciones vs resultados
# ===================================================================

def run_learning_module() -> str:
    """
    Lee el historial, consulta resultados reales del día y días anteriores
    pendientes, compara y genera un análisis de acierto/fallo.
    Devuelve el mensaje de resumen.
    """
    df = load_historial()
    if df.empty:
        return "El historial de predicciones está vacío. No hay nada que analizar."

    # Filtrar predicciones que aún no han sido evaluadas
    pending = df[
        (df["prediction_correct"].isna()) | (df["prediction_correct"] == "")
    ]

    if pending.empty:
        # Generar resumen general
        return _generate_summary(df)

    logger.info("Predicciones pendientes de evaluación: %d", len(pending))

    updated_count = 0
    for idx, row in pending.iterrows():
        sport_key = row["sport_key"]
        allsport_sport = SPORT_KEY_TO_ALLSPORT.get(sport_key)
        if not allsport_sport:
            logger.warning("Deporte no mapeado: %s", sport_key)
            continue

        # Obtener la fecha del evento
        try:
            event_dt = datetime.fromisoformat(
                str(row["commence_time"]).replace("Z", "+00:00")
            )
            event_date = event_dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            logger.warning("Fecha inválida para event_id=%s", row["event_id"])
            continue

        # Solo evaluar partidos que ya deberían haber terminado (al menos 3h después)
        now_utc = datetime.now(timezone.utc)
        if event_dt.tzinfo is None:
            event_dt = event_dt.replace(tzinfo=timezone.utc)
        if event_dt + timedelta(hours=3) > now_utc:
            continue  # Partido probablemente aún no terminó

        results = fetch_results(allsport_sport, event_date, event_date)

        for res in results:
            res_home = res.get("event_home_team", "")
            res_away = res.get("event_away_team", "")
            res_date = res.get("event_date", "")

            # Coincidencia por equipos y fecha
            if (
                res_home == row["home_team"]
                and res_away == row["away_team"]
                and res_date == event_date
            ):
                winner = determine_winner(res)
                if winner:
                    is_correct = str(row["team_predicted"]) == winner
                    df.at[idx, "actual_winner"] = winner
                    df.at[idx, "prediction_correct"] = is_correct
                    df.at[idx, "analysis_date"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    updated_count += 1
                break

    # Guardar historial actualizado
    df.to_csv(HISTORIAL_FILE, index=False)
    logger.info("Predicciones evaluadas en esta ejecución: %d", updated_count)

    # Generar resumen
    summary = _generate_summary(df)

    # Guardar análisis en archivo separado
    _save_analysis(df)

    return summary


def _generate_summary(df: pd.DataFrame) -> str:
    """Genera un resumen textual del rendimiento de las predicciones."""
    evaluated = df[df["prediction_correct"].notna() & (df["prediction_correct"] != "")]

    if evaluated.empty:
        return "Aún no hay predicciones evaluadas con resultados reales."

    total = len(evaluated)
    correct = evaluated["prediction_correct"].astype(str).str.lower().isin(["true", "1"]).sum()
    wrong = total - correct
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Desglose por deporte
    breakdown = []
    for sport in evaluated["sport_key"].unique():
        sport_df = evaluated[evaluated["sport_key"] == sport]
        s_total = len(sport_df)
        s_correct = sport_df["prediction_correct"].astype(str).str.lower().isin(["true", "1"]).sum()
        s_acc = (s_correct / s_total) * 100 if s_total > 0 else 0
        sport_name = sport.replace("_", " ").title()
        breakdown.append(f"  • {sport_name}: {s_correct}/{s_total} ({s_acc:.1f}%)")

    breakdown_text = "\n".join(breakdown)

    # Predicciones pendientes
    pending = df[
        (df["prediction_correct"].isna()) | (df["prediction_correct"] == "")
    ]

    msg = (
        f"📊 *Análisis de Predicciones*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Total evaluadas: {total}\n"
        f"✅ Aciertos: {correct}\n"
        f"❌ Fallos: {wrong}\n"
        f"📈 Precisión global: {accuracy:.1f}%\n\n"
        f"*Desglose por deporte:*\n{breakdown_text}\n\n"
        f"Predicciones pendientes: {len(pending)}"
    )
    return msg


def _save_analysis(df: pd.DataFrame) -> None:
    """Guarda un resumen del análisis en un CSV separado."""
    evaluated = df[df["prediction_correct"].notna() & (df["prediction_correct"] != "")]
    if evaluated.empty:
        return

    analysis_rows = []
    for sport in evaluated["sport_key"].unique():
        sport_df = evaluated[evaluated["sport_key"] == sport]
        total = len(sport_df)
        correct = sport_df["prediction_correct"].astype(str).str.lower().isin(["true", "1"]).sum()
        analysis_rows.append({
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sport_key": sport,
            "total_predictions": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": round((correct / total) * 100, 2) if total > 0 else 0,
        })

    df_analysis = pd.DataFrame(analysis_rows)

    # Append si ya existe
    if os.path.exists(ANALYSIS_FILE):
        df_existing = pd.read_csv(ANALYSIS_FILE)
        df_analysis = pd.concat([df_existing, df_analysis], ignore_index=True)

    df_analysis.to_csv(ANALYSIS_FILE, index=False)
    logger.info("Análisis guardado en %s", ANALYSIS_FILE)


# ===================================================================
# MÓDULO 7: Git — Commit y Push
# ===================================================================

def git_commit_and_push() -> None:
    """Hace commit y push de los archivos CSV actualizados al repositorio."""
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not GH_PAT or not repo:
        logger.warning(
            "GH_PAT o GITHUB_REPOSITORY no configurados. "
            "No se realizará commit/push."
        )
        return

    commands = [
        ["git", "config", "--global", "user.email", "actions@github.com"],
        ["git", "config", "--global", "user.name", "GitHub Actions Bot"],
        ["git", "add", HISTORIAL_FILE, ANALYSIS_FILE],
        ["git", "diff", "--cached", "--quiet"],  # Verificar si hay cambios
    ]

    for cmd in commands[:3]:
        subprocess.run(cmd, check=False, capture_output=True)

    # Verificar si hay cambios staged
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], capture_output=True
    )
    if result.returncode == 0:
        logger.info("No hay cambios para commitear.")
        return

    # Commit
    commit_result = subprocess.run(
        ["git", "commit", "-m", "Actualizar historial y análisis de predicciones"],
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        logger.error("Error en git commit: %s", commit_result.stderr)
        return

    # Push usando GH_PAT
    push_url = f"https://x-access-token:{GH_PAT}@github.com/{repo}.git"
    push_result = subprocess.run(
        ["git", "push", push_url, "HEAD:main"],
        capture_output=True,
        text=True,
    )
    if push_result.returncode == 0:
        logger.info("Push realizado exitosamente.")
    else:
        logger.error("Error en git push: %s", push_result.stderr)


# ===================================================================
# MÓDULO 8: Modos de ejecución principales
# ===================================================================

def run_futbol() -> None:
    """Modo Fútbol: obtiene cuotas y genera predicciones seguras."""
    logger.info("=" * 50)
    logger.info("MODO: FÚTBOL — Obteniendo predicciones seguras")
    logger.info("=" * 50)

    all_predictions = []
    for sport_key in FOOTBALL_SPORT_KEYS:
        events = fetch_odds(sport_key)
        if events:
            preds = filter_safe_predictions(events)
            all_predictions.extend(preds)

    if all_predictions:
        save_predictions(all_predictions)

        msg = "⚽ *Predicciones Seguras — Fútbol*\n━━━━━━━━━━━━━━━━━━━━\n"
        for p in all_predictions:
            league = p["sport_key"].replace("soccer_", "").replace("_", " ").title()
            prob_pct = round(p["implied_probability"] * 100, 1)
            msg += (
                f"\n🏟 *{p['home_team']}* vs *{p['away_team']}*\n"
                f"  Liga: {league}\n"
                f"  Predicción: *{p['team_predicted']}*\n"
                f"  Cuota: {p['odds']} | Prob: {prob_pct}%\n"
                f"  Casa: {p['bookmaker']}\n"
            )
        msg += f"\n_Total: {len(all_predictions)} predicciones seguras_"
        send_telegram(msg)
        git_commit_and_push()
    else:
        send_telegram(
            "⚽ No se encontraron predicciones seguras de Fútbol en este momento."
        )
    logger.info("Modo Fútbol completado.")


def run_mlb_nba() -> None:
    """Modo MLB/NBA: obtiene cuotas y genera predicciones seguras."""
    logger.info("=" * 50)
    logger.info("MODO: MLB/NBA — Obteniendo predicciones seguras")
    logger.info("=" * 50)

    all_predictions = []

    # MLB
    mlb_events = fetch_odds("baseball_mlb")
    if mlb_events:
        all_predictions.extend(filter_safe_predictions(mlb_events))

    # NBA
    nba_events = fetch_odds("basketball_nba")
    if nba_events:
        all_predictions.extend(filter_safe_predictions(nba_events))

    if all_predictions:
        save_predictions(all_predictions)

        msg = "⚾🏀 *Predicciones Seguras — MLB / NBA*\n━━━━━━━━━━━━━━━━━━━━\n"
        for p in all_predictions:
            sport_emoji = "⚾" if "mlb" in p["sport_key"] else "🏀"
            sport_label = "MLB" if "mlb" in p["sport_key"] else "NBA"
            prob_pct = round(p["implied_probability"] * 100, 1)
            msg += (
                f"\n{sport_emoji} *{p['home_team']}* vs *{p['away_team']}*\n"
                f"  Deporte: {sport_label}\n"
                f"  Predicción: *{p['team_predicted']}*\n"
                f"  Cuota: {p['odds']} | Prob: {prob_pct}%\n"
                f"  Casa: {p['bookmaker']}\n"
            )
        msg += f"\n_Total: {len(all_predictions)} predicciones seguras_"
        send_telegram(msg)
        git_commit_and_push()
    else:
        send_telegram(
            "⚾🏀 No se encontraron predicciones seguras de MLB/NBA en este momento."
        )
    logger.info("Modo MLB/NBA completado.")


def run_aprendizaje() -> None:
    """Modo Aprendizaje: compara predicciones con resultados reales."""
    logger.info("=" * 50)
    logger.info("MODO: APRENDIZAJE — Analizando resultados")
    logger.info("=" * 50)

    summary = run_learning_module()
    send_telegram(summary)
    git_commit_and_push()
    logger.info("Modo Aprendizaje completado.")


# ===================================================================
# PUNTO DE ENTRADA
# ===================================================================

def main():
    if len(sys.argv) < 2:
        print("Uso: python bot.py [futbol|mlb_nba|aprendizaje]")
        sys.exit(1)

    mode = sys.argv[1].lower().strip()

    modes = {
        "futbol": run_futbol,
        "fútbol": run_futbol,
        "mlb_nba": run_mlb_nba,
        "aprendizaje": run_aprendizaje,
    }

    handler = modes.get(mode)
    if handler:
        try:
            handler()
        except Exception as exc:
            error_msg = f"❌ Error crítico en modo '{mode}': {exc}"
            logger.exception(error_msg)
            send_telegram(error_msg)
            sys.exit(1)
    else:
        logger.error("Modo '%s' no reconocido.", mode)
        print(f"Modos disponibles: {', '.join(modes.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
