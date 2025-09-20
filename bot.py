# bot.py
# ProTrader Telegram Bot (manual-payments, Twelve Data signals, admin approval)
# Save this file exactly as bot.py
# Run: python bot.py
# NOTE: For production, move secrets to environment variables or Railway secrets.

import os
import io
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InputMediaPhoto,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

import matplotlib.pyplot as plt

# ---------------------------
# CONFIG: Secrets & Settings
# ---------------------------
# If you prefer safer setup, set these as env vars on Railway and they will be used.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "7790080100:AAGwX4riIDhZ9JKn6qnQ1UsDEa4EkNZSlE8"
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY") or "f7249b9c22574caea6d71ac931d3f8e0"

# Admin (your Telegram ID)
ADMIN_ID = int(os.getenv("ADMIN_ID") or 7239427141)

# Manual payment details (displayed to users)
PAYPAL_EMAIL = os.getenv("PAYPAL_EMAIL") or "susanzeedy4259@gmail.com"
MPESA_NUMBER = os.getenv("MPESA_NUMBER") or "0701767822"
BNB_ADDRESS = os.getenv("BNB_ADDRESS") or "0x412930bc47da7a7b5929ae8876ac41e7d39bc9e2"
USDT_TRON_ADDRESS = os.getenv("USDT_TRON_ADDRESS") or "TD6TWzH3NW9Phfws6DUDKkpgWLjf9924md"

# Packages definition
PACKAGES = {
    "starter": {"name": "Starter", "price_usd": 10, "days": 30, "description": "Major pairs, basic charts"},
    "pro": {"name": "Pro", "price_usd": 25, "days": 90, "description": "All forex + AI signals"},
    "vip": {"name": "VIP", "price_usd": 70, "days": 36500, "description": "Lifetime VIP access"},
}

# Database
DB_URL = os.getenv("DATABASE_URL") or "sqlite:///./forexbot.db"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Database models
# ---------------------------
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_premium = Column(Boolean, default=False)
    premium_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PaymentClaim(Base):
    __tablename__ = "payment_claims"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, index=True)
    package_key = Column(String)
    method = Column(String)  # "mpesa", "paypal", "crypto", "manual"
    details = Column(Text)   # txid, screenshot comment, etc
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)

# ---------------------------
# Utilities: DB helpers
# ---------------------------
def get_or_create_user(telegram_user):
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == telegram_user.id).first()
    if not user:
        user = User(
            telegram_id=telegram_user.id,
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    db.close()
    return user

def create_claim(telegram_id, package_key, method, details):
    db = SessionLocal()
    claim = PaymentClaim(
        telegram_id=telegram_id,
        package_key=package_key,
        method=method,
        details=details
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)
    db.close()
    return claim

def approve_latest_claim_for_user(telegram_id):
    db = SessionLocal()
    claim = db.query(PaymentClaim).filter(PaymentClaim.telegram_id == telegram_id, PaymentClaim.approved == False).order_by(PaymentClaim.created_at.desc()).first()
    if not claim:
        db.close()
        return False, None
    claim.approved = True
    claim.processed_at = datetime.utcnow()
    db.add(claim)
    # upgrade user
    user = db.query(User).filter(User.telegram_id == telegram_id).first()
    pkg = PACKAGES.get(claim.package_key, None)
    days = pkg["days"] if pkg else 30
    if user:
        now = datetime.utcnow()
        if not user.premium_until or user.premium_until < now:
            user.premium_until = now + timedelta(days=days)
        else:
            user.premium_until = user.premium_until + timedelta(days=days)
        user.is_premium = True
        db.add(user)
    db.commit()
    db.close()
    return True, claim.package_key

def reject_latest_claim_for_user(telegram_id):
    db = SessionLocal()
    claim = db.query(PaymentClaim).filter(PaymentClaim.telegram_id == telegram_id, PaymentClaim.approved == False).order_by(PaymentClaim.created_at.desc()).first()
    if not claim:
        db.close()
        return False
    db.delete(claim)
    db.commit()
    db.close()
    return True

# ---------------------------
# Market data: Twelve Data
# ---------------------------
def twelvedata_price(symbol: str) -> Optional[float]:
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": symbol, "apikey": TWELVE_DATA_KEY}
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        if "price" in j:
            return float(j["price"])
    except Exception as e:
        logger.exception("TD price error")
    return None

def twelvedata_candles(symbol: str, interval: str="1h", outputsize: int=200) -> List[dict]:
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "format": "JSON", "apikey": TWELVE_DATA_KEY}
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
        return j.get("values", [])
    except Exception:
        return []

def generate_signal_simple(symbol: str):
    # simple sma(20/50) logic using 1h candles
    candles = twelvedata_candles(symbol, interval="1h", outputsize=200)
    closes = [float(c["close"]) for c in reversed(candles)] if candles else []
    price = twelvedata_price(symbol) or (closes[-1] if closes else None)
    if not closes or len(closes) < 30:
        return {"pair": symbol, "side": "HOLD", "price": price, "sl": None, "tp": None, "confidence": 40}
    def sma(vals, n):
        vals = list(map(float, vals))
        if len(vals) < n:
            return sum(vals)/len(vals)
        return sum(vals[-n:]) / n
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    last = closes[-1]
    if s20 > s50:
        side = "BUY"
        sl = round(last * (1 - 0.0035), 6)
        tp = round(last * (1 + 0.006), 6)
        conf = 65
    elif s20 < s50:
        side = "SELL"
        sl = round(last * (1 + 0.0035), 6)
        tp = round(last * (1 - 0.006), 6)
        conf = 65
    else:
        side = "HOLD"
        sl = tp = None
        conf = 45
    return {"pair": symbol, "side": side, "price": price or last, "sl": sl, "tp": tp, "confidence": conf}

def plot_candles_to_buffer(symbol: str, interval="1h"):
    candles = twelvedata_candles(symbol, interval=interval, outputsize=200)
    if not candles:
        return None
    # get closes oldest->newest
    closes = [float(c["close"]) for c in reversed(candles)]
    times = [c["datetime"] for c in reversed(candles)]
    plt.figure(figsize=(8,3))
    plt.plot(closes[-120:], linewidth=1)
    plt.title(f"{symbol} - {interval}")
    plt.xlabel("bars")
    plt.ylabel("price")
    plt.grid(alpha=0.3)
    bio = io.BytesIO()
    plt.tight_layout()
    plt.savefig(bio, format="png")
    plt.close()
    bio.seek(0)
    return bio

# ---------------------------
# Telegram handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = get_or_create_user(update.effective_user)
    welcome = (
        "👋 *Welcome to Exness Global Forex AI (ProTrader)*\n\n"
        "I provide professional forex signals, charts and premium analysis.\n\n"
        "Free users get limited access. Choose from the menu below to begin."
    )
    keyboard = [["📊 Market"], ["🔥 Signals"], ["💼 My Profile"], ["💳 Upgrade"]]
    reply = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome, reply_markup=reply, parse_mode="Markdown")

async def market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # show some live prices
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    lines = []
    for p in pairs:
        price = twelvedata_price(p)
        lines.append(f"{p}: {price if price else 'N/A'}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="*Live Prices*\n"+ "\n".join(lines), parse_mode="Markdown")

async def signals_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == update.effective_user.id).first()
    db.close()
    now = datetime.utcnow()
    premium = False
    if user and user.is_premium and (not user.premium_until or user.premium_until > now):
        premium = True
    if not premium:
        # demo signal
        sig = generate_signal_simple("EURUSD")
        text = f"*Demo Signal* — {sig['pair']}\nSignal: {sig['side']}\nEntry: {sig['price']}\nSL: {sig['sl']}\nTP: {sig['tp']}\nConfidence: {sig['confidence']}%\n\nUpgrade for live multi-pair signals."
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
    else:
        pairs = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"]
        lines = ["*Premium Signals*"]
        for p in pairs:
            s = generate_signal_simple(p)
            lines.append(f"{s['pair']} | {s['side']} | Entry {s['price']} | SL {s['sl']} | TP {s['tp']} | {s['confidence']}%")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="\n".join(lines), parse_mode="Markdown")

async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_id == update.effective_user.id).first()
    db.close()
    if not user:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No profile found. Send /start to register.")
        return
    status = "Premium" if user.is_premium and (not user.premium_until or user.premium_until > datetime.utcnow()) else "Free"
    until = user.premium_until.isoformat() if user.premium_until else "N/A"
    txt = f"User: @{user.username or ''}\nID: {user.telegram_id}\nStatus: {status}\nPremium until: {until}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

async def upgrade_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = []
    for key, pkg in PACKAGES.items():
        buttons.append([InlineKeyboardButton(f"{pkg['name']} - ${pkg['price_usd']}", callback_data=f"pkg|{key}")])
    kb = InlineKeyboardMarkup(buttons)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Choose a package to purchase:", reply_markup=kb)

async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    user = query.from_user
    if data.startswith("pkg|"):
        _, key = data.split("|",1)
        pkg = PACKAGES.get(key)
        if not pkg:
            await query.edit_message_text("Package not found.")
            return
        # show payment options and instructions
        text = (
            f"*{pkg['name']}* — ${pkg['price_usd']}\n{pkg['description']}\n\n"
            "Payment options (manual):\n"
            f"• PayPal: `{PAYPAL_EMAIL}` (send payment as *{pkg['price_usd']} USD* and paste TXID)\n"
            f"• M-Pesa: `{MPESA_NUMBER}` (send and paste MPESA code / screenshot)\n"
            f"• Crypto: BNB `{BNB_ADDRESS}` or USDT(TRC20) `{USDT_TRON_ADDRESS}` (paste TXID)\n\n"
            "After you pay, send a message in this chat like:\n`I paid <method> <package> <details>`\nExample:\n`I paid mpesa starter STK12345`"
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
        return
    if data.startswith("approve|") or data.startswith("reject|"):
        # admin-only approve/reject (these buttons are only sent to admin)
        if user.id != ADMIN_ID:
            await query.edit_message_text("You are not authorized to perform this action.")
            return
        cmd, target = data.split("|",1)
        target_id = int(target)
        if cmd == "approve":
            ok, pkg = approve_latest_claim_for_user(target_id)
            if ok:
                await context.bot.send_message(chat_id=target_id, text="🎉 Your payment was approved. Premium activated. Enjoy pro signals!")
                await query.edit_message_text(f"Approved user {target_id} (pkg: {pkg})")
            else:
                await query.edit_message_text("No pending claim found to approve.")
        else:
            ok = reject_latest_claim_for_user(target_id)
            if ok:
                await context.bot.send_message(chat_id=target_id, text="❌ Your payment claim was rejected by admin.")
                await query.edit_message_text(f"Rejected user {target_id}")
            else:
                await query.edit_message_text("No pending claim found to reject.")
        return

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    user_obj = update.effective_user
    get_or_create_user(user_obj)  # ensures user exists in DB

    # quick commands mapping
    if text.lower().startswith("/start"):
        await start(update, context); return
    if text in ["📊 Market", "/market", "/forex"]:
        await market(update, context); return
    if text in ["🔥 Signals", "/signals", "/signal"]:
        await signals_cmd(update, context); return
    if text in ["💼 My Profile", "/profile"]:
        await profile(update, context); return
    if text in ["💳 Upgrade", "/upgrade"]:
        await upgrade_menu(update, context); return

    # Payment claim flow: "I paid <method> <package> <details>"
    # we'll accept flexible formats
    lower = text.lower()
    if lower.startswith("i paid") or lower.startswith("paid"):
        # parse loosely:
        parts = text.split(maxsplit=3)
        # try to find method and package
        method = None
        package_key = None
        details = text
        tokens = text.lower().split()
        for tk in tokens:
            if tk in ["mpesa", "m-pesa", "mpesa:"]:
                method = "mpesa"
            if tk in ["paypal", "pp"]:
                method = "paypal"
            if tk in ["crypto", "bnb", "usdt", "trx", "btc"]:
                method = "crypto"
        for key, pkg in PACKAGES.items():
            if key in text.lower() or pkg['name'].lower() in text.lower():
                package_key = key
                break
        if not package_key:
            package_key = "starter"
        # details = everything after the method/package if possible
        if len(parts) >= 3:
            details = " ".join(parts[2:]) if parts[0].lower() in ["i","paid"] else " ".join(parts[1:])
        # create claim
        create_claim(update.effective_user.id, package_key, method or "manual", details)
        # notify admin with approve/reject buttons
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("Approve ✅", callback_data=f"approve|{update.effective_user.id}"),
            InlineKeyboardButton("Reject ❌", callback_data=f"reject|{update.effective_user.id}")
        ]])
        admin_text = (
            f"🔔 *Payment claim*\nUser: @{update.effective_user.username or ''} ({update.effective_user.id})\n"
            f"Package: {PACKAGES[package_key]['name']}\nMethod: {method}\nDetails: {details}\n\nApprove or reject."
        )
        await context.bot.send_message(chat_id=ADMIN_ID, text=admin_text, reply_markup=kb, parse_mode="Markdown")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Thanks — your claim has been submitted. Admin will review and approve shortly.")
        return

    # chart request: "chart EURUSD 1h"
    if lower.startswith("chart"):
        # format: chart EURUSD 1h
        parts = text.split()
        if len(parts) >= 2:
            symbol = parts[1].upper()
            interval = parts[2] if len(parts) >= 3 else "1h"
            bio = plot_candles_to_buffer(symbol, interval)
            if bio:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=bio)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Chart not available for that symbol.")
            return

    # help fallback
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I didn't understand. Use the menu: /start, /upgrade, /signals, /profile")

# ---------------------------
# Main: build and run
# ---------------------------
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # command handlers
    app.add_handler(CommandHandler("start", start))
    # message handlers (buttons send plain text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # callback queries for inline buttons
    app.add_handler(CallbackQueryHandler(callback_router))

    # also provide explicit command handlers for convenience
    app.add_handler(CommandHandler("market", market))
    app.add_handler(CommandHandler("signals", signals_cmd))
    app.add_handler(CommandHandler("profile", profile))
    app.add_handler(CommandHandler("upgrade", upgrade_menu))

    logger.info("Bot starting...")
    app.run_polling(allowed_updates=["message", "callback_query"])

if __name__ == "__main__":
    main()