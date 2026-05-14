from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from datetime import datetime
import os
import json
import joblib
import numpy as np


def ensure_secret_key(app: Flask) -> None:
    secret = os.environ.get("FLASK_SECRET_KEY") or "dev-secret-inno-stay"
    app.secret_key = secret


def month_to_season(month: int) -> str:
    if month in [4, 5, 6]:
        return "summer"
    if month in [7, 8, 9]:
        return "rainy"
    if month == 12:
        return "festive"
    return "normal"


def is_weekend(date_str: str) -> bool:
    try:
        # Expecting YYYY-MM-DD
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() >= 5
    except Exception:
        # If parsing fails, fall back to False (weekday adjustment)
        return False


def load_model(model_path: str):
    if not os.path.exists(model_path):
        # Train on-the-fly if not present
        from model.train_model import train_and_save_model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        train_and_save_model(model_path)
    return joblib.load(model_path)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    ensure_secret_key(app)

    model_path = os.path.join("model", "model.pkl")
    model = load_model(model_path)

    @app.route("/")
    def root():
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email")
            # password is accepted but not stored for this demo
            if email:
                user = session.get("user", {})
                user["email"] = email
                session["user"] = user
                return redirect(url_for("user_details"))
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/user_details", methods=["GET", "POST"])
    def user_details():
        if request.method == "POST":
            # Store user profile and preferences
            name = request.form.get("name")
            mobile = request.form.get("mobile")
            city = request.form.get("city", "")
            gender = request.form.get("gender")
            address = request.form.get("address")
            pincode = request.form.get("pincode")
            user = session.get("user", {})
            if name:
                user["name"] = name
            if mobile:
                user["mobile"] = mobile
            if gender:
                user["gender"] = gender
            if address:
                user["address"] = address
            if pincode:
                user["pincode"] = pincode
            session["user"] = user

            session["prefs"] = {
                "city": city,
            }
            return redirect(url_for("hotel_search"))
        return render_template("user_details.html", user=session.get("user"))

    @app.route('/settings', methods=['GET'])
    def settings():
        import random
        user = session.get('user', {})
        # contact number: random Indian mobile-like number for demo
        contact_number = f"+91-9{random.randint(10_000_000, 99_999_999)}"
        bookings = session.get('bookings', [])
        help_requests = session.get('help_requests', [])
        theme = session.get('theme', 'light')
        now_str = session.get('settings_last_updated') or datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        return render_template('settings.html', user=user, contact_number=contact_number, bookings=bookings, help_requests=help_requests, theme=theme, now_str=now_str)

    @app.route('/settings', methods=['POST'])
    def settings_save_profile():
        # Save profile settings: name, email, contact, password (password stored only in session for demo)
        name = request.form.get('name')
        email = request.form.get('email')
        contact = request.form.get('contact')
        password = request.form.get('password')
        user = session.get('user', {})
        if name:
            user['name'] = name
        if email:
            user['email'] = email
        if contact:
            user['mobile'] = contact
        if password:
            user['password'] = password
        session['user'] = user
        return redirect(url_for('settings'))

    @app.route('/settings/notifications', methods=['POST'])
    def settings_notifications():
        data = request.get_json(silent=True) or {}
        # Expected keys: price_drop, weekend_offers, new_recommendations (true/false)
        notifications = {
            'price_drop': bool(data.get('price_drop')),
            'weekend_offers': bool(data.get('weekend_offers')),
            'new_recommendations': bool(data.get('new_recommendations')),
        }
        session['notifications'] = notifications
        return jsonify({'status': 'ok', 'notifications': notifications})

    @app.route('/settings/privacy', methods=['POST'])
    def settings_privacy():
        action = request.form.get('action') or (request.get_json(silent=True) or {}).get('action')
        if not action:
            return jsonify({'status': 'error', 'message': 'action required'}), 400
        if action == 'clear_history':
            session.pop('history', None)
            session.pop('bookings', None)
            return jsonify({'status': 'ok', 'message': 'History cleared'})
        if action == 'delete_data':
            # Clear user data (demo)
            session.pop('user', None)
            session.pop('prefs', None)
            session.pop('history', None)
            session.pop('bookings', None)
            session.pop('help_requests', None)
            session.pop('notifications', None)
            return jsonify({'status': 'ok', 'message': 'All saved data deleted'})
        return jsonify({'status': 'error', 'message': 'unknown action'}), 400

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/contact', methods=['GET', 'POST'])
    def contact():
        if request.method == 'POST':
            # accept form data
            name = request.form.get('name')
            email = request.form.get('email')
            message = request.form.get('message')
            contacts = session.get('contacts', [])
            entry = {'name': name, 'email': email, 'message': message, 'timestamp': datetime.utcnow().isoformat() + 'Z'}
            contacts.insert(0, entry)
            session['contacts'] = contacts[:50]
            return jsonify({'status': 'ok', 'entry': entry})
        return render_template('contact.html')

    @app.route('/settings/help', methods=['POST'])
    def settings_help():
        # Accept form-encoded or JSON payloads safely
        question = None
        try:
            if request.content_type and 'application/json' in request.content_type:
                payload = request.get_json(silent=True) or {}
                question = payload.get('question')
            else:
                question = request.form.get('question')
        except Exception:
            question = None

        if not question:
            return jsonify({'status': 'error', 'message': 'Question required'}), 400
        help_requests = session.get('help_requests', [])
        entry = {'question': question, 'timestamp': datetime.utcnow().isoformat() + 'Z'}
        help_requests.insert(0, entry)
        session['help_requests'] = help_requests[:20]
        return jsonify({'status': 'ok', 'entry': entry})

    @app.route('/settings/theme', methods=['POST'])
    def settings_theme():
        data = request.get_json(silent=True) or {}
        theme = data.get('theme')
        if theme not in ('light', 'dark'):
            return jsonify({'status': 'error', 'message': 'Invalid theme'}), 400
        session['theme'] = theme
        return jsonify({'status': 'ok', 'theme': theme})

    @app.route("/search")
    def hotel_search():
        return render_template("search.html")

    def compute_price_with_breakdown(base_price: float, payload: dict):
        amenities = payload.get("amenities", {})
        checkin_month = int(payload.get("checkin_month", 1))
        checkin_date = payload.get("checkin_date") or payload.get("checkin", "")

        # Amenities adjustments (simple additive per amenity)
        amenity_adders = {
            "wifi": 150,
            "ac": 200,
            "breakfast": 250,
            "pool": 300,
            "parking": 120,
            "gym": 180,
            "spa": 400,
            "bar": 220,
            "room_service": 160,
        }
        amenity_total = 0
        amenities_detail = {}
        for key, add in amenity_adders.items():
            if amenities.get(key, False):
                amenity_total += add
                amenities_detail[key] = add

        season = month_to_season(checkin_month)
        seasonal_pct = 0.0
        seasonal_label = None
        if season == "summer":
            seasonal_pct = 0.20
            seasonal_label = "Summer Special Offer (+20%)"
        elif season == "rainy":
            seasonal_pct = -0.15
            seasonal_label = "Rainy Discount (-15%)"
        elif season == "festive":
            seasonal_pct = 0.25
            seasonal_label = "Festive Surge (+25%)"

        weekend_flag = is_weekend(checkin_date)
        weekday_weekend_pct = 0.05 if weekend_flag else -0.10
        weekday_weekend_label = "+5% Weekend" if weekend_flag else "-10% Weekday"

        adjusted = base_price
        base_component = round(base_price, 2)
        adjusted += amenity_total
        amenities_component = round(amenity_total, 2)
        adjusted *= (1 + seasonal_pct)
        seasonal_component = f"{int(seasonal_pct * 100)}%"
        adjusted *= (1 + weekday_weekend_pct)
        weekday_weekend_component = f"{int(weekday_weekend_pct * 100)}%"

        final_price = max(999, round(adjusted, 2))

        breakdown = {
            "base": base_component,
            "amenities": amenities_component,
            "amenities_detail": amenities_detail,
            "season": {
                "label": seasonal_label or "Normal Season (0%)",
                "percent": seasonal_component,
            },
            "weekday_weekend": {
                "label": weekday_weekend_label,
                "percent": weekday_weekend_component,
            },
        }
        return final_price, breakdown

    def suggest_hotels(payload: dict, target_price: float):
        city = payload.get("city", "Mumbai")
        star = int(payload.get("star_rating", 4))
        room_type = payload.get("room_type", "Standard")
        hotel_type = payload.get("hotel_type", "Veg")
        amenities = payload.get("amenities", {})

        sample_hotels = [
            {
                "id": "H1001",
                "name": "Crimson Orchid",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.2)),
                "price": max(1200, round(target_price * 0.95)),
                "image": "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1200",
            },
            {
                "id": "H1002",
                "name": "Ruby Residency",
                "city": city,
                "rating": min(5.0, max(3.5, star - 0.1)),
                "price": max(1100, round(target_price * 1.0)),
                "image": "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?w=1200",
            },
            {
                "id": "H1003",
                "name": "Rich Stay",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.4)),
                "price": max(1300, round(target_price * 1.05)),
                "image": "https://images.unsplash.com/photo-1501117716987-c8e2a3a67c73?w=1200",
            },
            {
                "id": "H1004",
                "name": "Vermilion Vista",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.0)),
                "price": max(1000, round(target_price * 0.9)),
                "image": "https://images.unsplash.com/photo-1528909514045-2fa4ac7a08ba?w=1200",
            },
            {
                "id": "H1005",
                "name": "Red Maple Inn",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.3)),
                "price": max(1250, round(target_price * 0.98)),
                "image": "https://images.unsplash.com/photo-1618773928121-c32242e63f39?w=1200",
            },
            {
                "id": "H1006",
                "name": "Cardinal Crown Hotel",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.1)),
                "price": max(1150, round(target_price * 1.02)),
                "image": "https://images.unsplash.com/photo-1566665797739-1674de7a421a?w=1200",
            },
            {
                "id": "H1007",
                "name": "Crimson Bay Suites",
                "city": city,
                "rating": min(5.0, max(3.5, star + 0.5)),
                "price": max(1350, round(target_price * 1.08)),
                "image": "https://images.unsplash.com/photo-1611892440504-42a792e24d32?w=1200",
            },
        ]

        def icon_list(am: dict):
            return [k for k, v in am.items() if v]

        for h in sample_hotels:
            h.update({
                "room_type": room_type,
                "hotel_type": hotel_type,
                "amenities": icon_list(amenities),
            })

        # Pick up to 7
        hotels = sample_hotels[:7]
        avg_price = float(np.mean([h["price"] for h in hotels])) if hotels else target_price
        market_delta_pct = round(((target_price - avg_price) / max(1.0, avg_price)) * 100.0, 2)
        return hotels, avg_price, market_delta_pct

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)

        # Simple feature vector build (must match trainer)
        # Features: [city_idx, star_rating, guests, nights, month, room_type_idx, hotel_type_idx, amenity_count]
        def cat_index(value: str, options: list[str]) -> int:
            v = (value or "").strip().lower()
            for i, opt in enumerate(options):
                if v == opt:
                    return i
            return 0

        city_idx = cat_index(data.get("city"), [
            "mumbai", "delhi", "bangalore", "hyderabad", "chennai", "pune", "kolkata"
        ])
        room_type_idx = cat_index(data.get("room_type"), ["standard", "deluxe", "suite"])
        hotel_type_idx = cat_index(data.get("hotel_type"), ["veg", "non-veg"])  # display only
        star_rating = float(data.get("star_rating", 4))
        guests = float(data.get("guests", 2))
        nights = float(data.get("nights", 1))
        month = float(data.get("checkin_month", 1))

        amenities = data.get("amenities", {})
        amenity_count = float(sum(1 for _k, v in amenities.items() if v))

        X = np.array([[city_idx, star_rating, guests, nights, month, room_type_idx, hotel_type_idx, amenity_count]])

        base_pred = float(model.predict(X)[0])
        final_price, breakdown = compute_price_with_breakdown(base_pred, data)

        suggested, avg_price, market_delta_pct = suggest_hotels(data, final_price)

        # Store history
        history = session.get("history", [])
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "inputs": data,
            "predicted_price": final_price,
            "avg_price": avg_price,
            "market_delta_pct": market_delta_pct,
        }
        history.insert(0, entry)
        session["history"] = history[:10]

        return jsonify({
            "predicted_price": final_price,
            "breakdown": breakdown,
            "suggested_hotels": suggested,
            "avg_price": avg_price,
            "market_delta_pct": market_delta_pct,
        })

    @app.route("/history", methods=["GET"])
    def history():
        return jsonify(session.get("history", []))

    @app.route("/hotel/<hotel_id>")
    def hotel_detail(hotel_id: str):
        # In a real system fetch from DB; use session last suggested for now
        last = session.get("history", [])
        hotels = last[0].get("suggested_hotels", []) if last else []
        hotel = next((h for h in hotels if h.get("id") == hotel_id), None)
        if not hotel and hotels:
            hotel = hotels[0]
        # Fallback hotel
        hotel = hotel or {
            "id": hotel_id,
            "name": "InnoStay Hotel",
            "city": "",
            "rating": 4.2,
            "price": 2499,
            "image": "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=1600",
            "amenities": ["wifi", "ac", "breakfast"],
        }
        return render_template("hotel_detail.html", hotel=hotel)

    @app.route("/book", methods=["POST"])
    def book():
        data = request.get_json(force=True)
        hotel_id = data.get("hotel_id")
        user = session.get("user", {})
        booking_id = f"BK-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hotel_id or 'H'}"

        # Store lightweight booking confirmation in session
        bookings = session.get("bookings", [])
        # Find hotel price from last suggestions
        last = session.get("history", [])
        hotels = last[0].get("suggested_hotels", []) if last else []
        hotel = next((h for h in hotels if h.get("id") == hotel_id), None)
        price = hotel["price"] if hotel else 2499
        booking = {
            "booking_id": booking_id,
            "hotel_id": hotel_id,
            "user": user,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "amount": price,
            "payment": {
                "account_number": None,
                "user_name": None,
                "upi_transaction": None,
                "status": "pending"
            }
        }
        bookings.insert(0, booking)
        session["bookings"] = bookings[:20]

        return jsonify({
            "status": "confirmed",
            "booking_id": booking_id,
            "message": "Booking confirmed",
            "payment_required": True
        })


    @app.route("/payment/<booking_id>", methods=["GET", "POST"])
    def payment(booking_id):
        bookings = session.get("bookings", [])
        booking = next((b for b in bookings if b["booking_id"] == booking_id), None)
        if not booking:
            return "Booking not found", 404
        # Ensure booking has an amount to pay. Prefer booking.amount, else use latest prediction from history.
        last_history = session.get('history', [])
        # predicted_price from most recent prediction (if available)
        predicted = None
        if last_history:
            predicted = last_history[0].get('predicted_price') or last_history[0].get('avg_price')
        if predicted is not None and not booking.get('predicted_price'):
            booking['predicted_price'] = predicted
        if not booking.get('amount'):
            # booked amount falls back to predicted when hotel price is missing
            booking['amount'] = booking.get('amount') or booking.get('predicted_price') or 0
        if request.method == "POST":
            account_number = request.form.get("account_number")
            user_name = request.form.get("user_name")
            upi_transaction = request.form.get("upi_transaction")
            booking["payment"] = {
                "account_number": account_number,
                "user_name": user_name,
                "upi_transaction": upi_transaction,
                "status": "completed"
            }
            session["bookings"] = bookings
            return render_template("payment_complete.html", booking=booking)
        return render_template("payment_form.html", booking=booking)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)



