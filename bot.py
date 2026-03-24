import os
import io
import math
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ─────────────────────────────────────────────
# CONFIGURACION DE LOGS
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot_inversion.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. CONEXION
# ─────────────────────────────────────────────
api_key    = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

if not api_key or not secret_key:
    log.error("Faltan las variables de entorno ALPACA_API_KEY / ALPACA_SECRET_KEY.")
    exit(1)

cliente_trading = TradingClient(api_key, secret_key, paper=True)
cliente_datos   = StockHistoricalDataClient(api_key, secret_key)

# ─────────────────────────────────────────────
# 2. SENTIMIENTO DE NOTICIAS
# ─────────────────────────────────────────────
def obtener_nota_prensa(ticker: str) -> float:
    """
    Devuelve la media de polaridad TextBlob de los 5 últimos titulares.
    Rango: -1.0 (muy negativo) → +1.0 (muy positivo). 0.0 si falla.
    """
    try:
        url = f"https://data.alpaca.markets/v1beta1/news?symbols={ticker}&limit=5"
        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
        res = requests.get(url, headers=headers, timeout=10).json()

        noticias = res.get('news', [])
        if not noticias:
            return 0.0

        scores = [TextBlob(n['headline']).sentiment.polarity for n in noticias]
        return sum(scores) / len(scores)
    except Exception as e:
        log.warning(f"[{ticker}] Error leyendo noticias: {e}")
        return 0.0

# ─────────────────────────────────────────────
# 3. CAJA Y STOP LOSS / TAKE PROFIT
# ─────────────────────────────────────────────
try:
    cuenta            = cliente_trading.get_account()
    posiciones        = cliente_trading.get_all_positions()
    posiciones_actuales = {p.symbol: p for p in posiciones}

    log.info(f"Efectivo disponible: {float(cuenta.cash):.2f} $")
    log.info("Revisando paracaídas de posiciones abiertas...")

    for pos in posiciones:
        ganancia_pct = float(pos.unrealized_plpc)
        simbolo      = pos.symbol

        if ganancia_pct >= 0.08:
            cliente_trading.close_position(simbolo)
            log.info(f"TAKE PROFIT ✅ {simbolo} cerrado con +{ganancia_pct*100:.2f}%")
            posiciones_actuales.pop(simbolo, None)

        elif ganancia_pct <= -0.03:
            cliente_trading.close_position(simbolo)
            log.info(f"STOP LOSS 🛑 {simbolo} cortado con {ganancia_pct*100:.2f}%")
            posiciones_actuales.pop(simbolo, None)

    # MEJORA: refrescar el cash DESPUÉS de cerrar posiciones
    cuenta      = cliente_trading.get_account()
    dinero_real = float(cuenta.cash)
    log.info(f"Efectivo tras ajustes: {dinero_real:.2f} $")

except Exception as e:
    log.error(f"Error al revisar caja/posiciones: {e}")
    exit(1)

presupuesto_total = dinero_real * 0.80   # usamos el 80% del cash disponible

# ─────────────────────────────────────────────
# 4. UNIVERSO DE ACTIVOS (S&P 500)
# ─────────────────────────────────────────────
SECTORES_MAX = 2   # máximo de activos por sector en la selección final

try:
    url    = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    header = {'User-Agent': 'Mozilla/5.0'}
    res    = requests.get(url, headers=header, timeout=15).text
    tabla  = pd.read_html(io.StringIO(res))[0]

    # Limpiamos símbolos problemáticos y guardamos sector
    tabla = tabla[~tabla['Symbol'].str.contains(r'[.\-]', regex=True)]
    simbolos_df = tabla[['Symbol', 'GICS Sector']].head(100)
    simbolos    = simbolos_df['Symbol'].tolist()
    sector_map  = dict(zip(simbolos_df['Symbol'], simbolos_df['GICS Sector']))

    log.info(f"Universo cargado: {len(simbolos)} activos del S&P 500")

except Exception as e:
    log.error(f"Error cargando universo desde Wikipedia: {e}")
    exit(1)

# ─────────────────────────────────────────────
# 5. DESCARGA DE HISTÓRICO (2 AÑOS)
# ─────────────────────────────────────────────
params = StockBarsRequest(
    symbol_or_symbols=simbolos,
    timeframe=TimeFrame.Day,
    start=datetime.now() - timedelta(days=730),
    end=datetime.now(),
    feed=DataFeed.IEX
)

try:
    velas      = cliente_datos.get_stock_bars(params)
    df_mercado = velas.df
    log.info("Histórico descargado correctamente.")
except Exception as e:
    log.error(f"No se pudieron obtener datos de mercado: {e}")
    exit(0)

# ─────────────────────────────────────────────
# 6. FILTRO DE VOLUMEN CON CONFIRMACION DIRECCIONAL
# ─────────────────────────────────────────────
calientes = []

for ticker in simbolos:
    try:
        if ticker not in df_mercado.index:
            continue

        d = df_mercado.loc[ticker]
        if len(d) < 200:
            continue

        media_vol   = d['volume'][:-1].mean()
        vol_hoy     = d['volume'].iloc[-1]
        cierre_hoy  = d['close'].iloc[-1]
        cierre_ayer = d['close'].iloc[-2]

        # MEJORA: volumen alto + precio subiendo (confirmación direccional)
        if media_vol > 0 and vol_hoy > media_vol * 1.5 and cierre_hoy > cierre_ayer:
            calientes.append({
                'Activo': ticker,
                'Ratio':  vol_hoy / media_vol,
                'Sector': sector_map.get(ticker, 'Desconocido')
            })
    except Exception:
        continue

if not calientes:
    log.info("Mercado tranquilo hoy. Sin candidatos por volumen+dirección. Saliendo.")
    exit(0)

# ─────────────────────────────────────────────
# 7. DIVERSIFICACION POR SECTOR
# ─────────────────────────────────────────────
df_calientes = (
    pd.DataFrame(calientes)
    .sort_values('Ratio', ascending=False)
)

seleccionados = []
conteo_sector = {}

for _, fila in df_calientes.iterrows():
    sector = fila['Sector']
    if conteo_sector.get(sector, 0) < SECTORES_MAX:
        seleccionados.append(fila['Activo'])
        conteo_sector[sector] = conteo_sector.get(sector, 0) + 1
    if len(seleccionados) >= 5:
        break

top_picks = seleccionados
log.info(f"Candidatos finales (diversificados): {top_picks}")

# ─────────────────────────────────────────────
# 8. MODELO ML CON TRAIN/TEST CORRECTO
# ─────────────────────────────────────────────
dinero_por_accion = presupuesto_total / len(top_picks) if top_picks else 0

for activo in top_picks:
    if activo in posiciones_actuales:
        log.info(f"[{activo}] Ya tenemos posición abierta, saltando.")
        continue

    try:
        datos = df_mercado.loc[activo].copy()

        # Features técnicos
        datos['retorno']     = datos['close'].pct_change()
        datos['m5']          = datos['close'].rolling(5).mean()
        datos['m20']         = datos['close'].rolling(20).mean()
        datos['volatilidad'] = datos['retorno'].rolling(10).std()

        delta    = datos['close'].diff()
        ganancia = delta.where(delta > 0, 0).rolling(14).mean()
        perdida  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs       = ganancia / (perdida + 1e-9)   # evitar división por cero
        datos['rsi'] = 100 - (100 / (1 + rs))

        # Target: ¿sube el precio mañana?
        datos['obj'] = (datos['close'].pct_change().shift(-1) > 0).astype(int)

        datos = datos.dropna()
        if len(datos) < 200:
            log.warning(f"[{activo}] Histórico insuficiente tras dropna, saltando.")
            continue

        features = ['m5', 'm20', 'retorno', 'volatilidad', 'rsi']
        X = datos[features]
        y = datos['obj']

        # MEJORA: split train/test real (80/20), sin incluir la última fila en entrenamiento
        corte = int(len(datos) * 0.80)
        X_train, X_test = X.iloc[:corte],    X.iloc[corte:-1]
        y_train, y_test = y.iloc[:corte],    y.iloc[corte:-1]
        X_pred          = X.iloc[[-1]]       # fila actual para predecir

        if len(X_train) < 100 or len(X_test) < 10:
            log.warning(f"[{activo}] Datos insuficientes para train/test, saltando.")
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Validación mínima: solo actuamos si el modelo supera el 55% en test
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc < 0.55:
            log.info(f"[{activo}] Modelo descartado (accuracy test: {acc:.2%})")
            continue

        pred          = model.predict(X_pred)[0]
        precio_actual = datos['close'].iloc[-1]

        log.info(f"[{activo}] Predicción: {'SUBE' if pred==1 else 'BAJA'} | Accuracy test: {acc:.2%}")

        if pred == 1:
            nota_prensa = obtener_nota_prensa(activo)

            if nota_prensa < -0.2:
                log.info(f"[{activo}] FRENADO por noticias negativas (score: {nota_prensa:.2f})")
            else:
                cantidad = math.floor(dinero_por_accion / precio_actual)
                if cantidad > 0:
                    cliente_trading.submit_order(MarketOrderRequest(
                        symbol=activo,
                        qty=cantidad,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    ))
                    log.info(
                        f"COMPRA 🟢 {activo} | {cantidad} acc. @ {precio_actual:.2f}$ "
                        f"| Prensa: {nota_prensa:.2f} | Acc: {acc:.2%}"
                    )
                else:
                    log.info(f"[{activo}] Presupuesto insuficiente para comprar 1 acción.")

    except Exception as e:
        log.error(f"[{activo}] Error en análisis: {e}")
        continue

log.info("─── Jornada terminada. Todo bajo control, jefe. ───")
