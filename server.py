import json, math, random, string, uuid, pathlib, csv
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- Setup ----------
ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

# ---------- Data ----------
def load_all_places():
    """
    Läser alla *.csv i /data enligt ditt schema:
    id,display_name,alt_names,street,postnummer,ort,kommun,lan,lat,lon,svardighet

    - Hanterar UTF-8 med/utan BOM
    - Tillåter både kommatecken och semikolon som avskiljare
    - Tillåter decimalkomma i lat/lon
    - Sätter city från filnamnet: stockholm/malmo/goteborg -> Stockholm/Malmö/Göteborg
    - Hoppar över rader utan giltiga koordinater
    """
    import re

    def coerce_float(val):
        if val is None:
            raise ValueError("None")
        s = str(val).strip()
        if not s:
            raise ValueError("empty")
        s = s.replace(",", ".")
        return float(s)

    def city_from_stem(stem: str) -> str:
        s = stem.strip().lower()
        if s == "malmo":   return "Malmö"
        if s == "goteborg": return "Göteborg"
        if s == "stockholm": return "Stockholm"
        # fallback: versal första bokstaven
        return s[:1].upper() + s[1:]

    places = []
    for p in DATA_DIR.glob("*.csv"):
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                # tillåt , eller ;
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                except Exception:
                    class _D: delimiter = ","
                    dialect = _D()

                reader = csv.DictReader(f, dialect=dialect)
                if not reader.fieldnames:
                    print(f"[ERROR] {p.name}: Saknar header.")
                    continue
                # normalisera headrar
                reader.fieldnames = [h.strip() for h in reader.fieldnames]

                required = {"lat", "lon"}
                missing = required - set(h.lower() for h in reader.fieldnames)
                if missing:
                    print(f"[WARN] {p.name}: saknar {missing} – försöker ändå om snarlika namn finns.")

                skipped = 0
                stem_city = city_from_stem(p.stem)

                for i, row in enumerate(reader, start=2):  # start=2 pga header på rad 1
                    # trimma blanksteg
                    row = { (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items() }

                    try:
                        lat = coerce_float(row.get("lat"))
                        lon = coerce_float(row.get("lon"))
                    except Exception as e:
                        skipped += 1
                        # Visa kort orsak, ofta "" (tom sträng) eller decimalkomma fel
                        print(f"[WARN] Skippar {p.name} rad {i}: ogiltig lat/lon ({e}).")
                        continue

                    # Fält enligt ditt schema + standardiserade fält
                    out = dict(row)
                    out["lat"] = lat
                    out["lon"] = lon
                    out["city"] = row.get("city") or stem_city
                    out["display_name"] = row.get("display_name") or row.get("id") or "Okänt"
                    places.append(out)

                print(f"[INFO] Läste {p.name}: {i-1-skipped} rader, {skipped} hoppade över.")

        except Exception as e:
            print(f"[ERROR] Kunde inte läsa {p.name}: {e}")

    return places

PLACES = load_all_places()
print(f"[INFO] Totalt platser: {len(PLACES)}")
if PLACES:
    counts = {}
    for p in PLACES:
        counts[p["city"]] = counts.get(p["city"], 0) + 1
    print("[INFO] Banor:", ", ".join(f"{k} ({v})" for k,v in counts.items()))



def list_tracks() -> List[str]:
    # Stabil, snygg ordning om de finns
    prefer = ["Stockholm", "Malmö", "Göteborg"]
    have = sorted({p["city"] for p in PLACES})
    # sortera så prefererade hamnar först
    ordered = [t for t in prefer if t in have] + [t for t in have if t not in prefer]
    return ordered

def list_tracks() -> List[str]:
    prefer = ["Stockholm", "Malmö", "Göteborg"]
    have = sorted({p["city"] for p in PLACES})
    ordered = [t for t in prefer if t in have] + [t for t in have if t not in prefer]
    return ordered

def places_for_track(track: str) -> List[dict]:
    # <--- DENNA SAKNAS HOS DIG
    return [p for p in PLACES if p.get("city","").lower() == track.lower()]



# ---------- Game logic ----------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    from math import radians, sin, cos, atan2, sqrt
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(math.sqrt(a), math.sqrt(1-a))

def score_from_distance_m(d: float) -> int:
    # 0–5000 m ger 5000–0 p linjärt (tweaka efter smak)
    MAX = 5000.0
    return int(round(MAX * max(0.0, 1.0 - d/5000.0)))

class CreateRoomIn(BaseModel):
    track: str
    rounds: int = 5
    time_limit: int = 60

def make_room_code(prefix="RM") -> str:
    import random, string
    return f"{prefix}-{''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(4))}"

class Player:
    def __init__(self, name: str, ws: WebSocket):
        self.name = name[:24] or "Spelare"
        self.ws = ws
        self.score_total = 0
        self.last_guess = None  # {"lat","lon","dist","score"}

class Room:
    def __init__(self, track: str, rounds: int, time_limit: int, host_token: str):
        self.id = make_room_code(prefix=track[:3].upper())
        self.track = track
        self.rounds = rounds
        self.time_limit = time_limit
        self.host_token = host_token
        self.players: Dict[str, Player] = {}
        self.state = "lobby"     # lobby|playing|ended
        self.round_index = -1
        self.targets: List[dict] = []
        self.current_target: Optional[dict] = None

    def pick_targets(self):
        pool = places_for_track(self.track)
        if len(pool) < self.rounds:
            raise ValueError("För få platser i banan.")
        import random
        self.targets = random.sample(pool, self.rounds)

    def to_public(self) -> dict:
        return {
            "room_id": self.id,
            "track": self.track,
            "rounds": self.rounds,
            "time_limit": self.time_limit,
            "state": self.state,
            "round_index": self.round_index,
            "players": [{"name": p.name, "score_total": p.score_total} for p in self.players.values()],
        }

ROOMS: Dict[str, Room] = {}

# ---------- REST ----------
@app.get("/api/tracks")
def api_tracks():
    tracks = list_tracks()
    return {"tracks": tracks, "counts": {t: len(places_for_track(t)) for t in tracks}}

@app.post("/api/room")
def api_create_room(req: CreateRoomIn):
    if req.track not in list_tracks():
        raise HTTPException(400, f"Saknar bana: {req.track}")
    room = Room(req.track, req.rounds, req.time_limit, host_token=uuid.uuid4().hex)
    room.pick_targets()
    ROOMS[room.id] = room
    return {"room_id": room.id, "host_token": room.host_token, "room": room.to_public()}

@app.get("/api/room/{room_id}")
def api_get_room(room_id: str):
    room = ROOMS.get(room_id)
    if not room: raise HTTPException(404, "Rum saknas")
    return room.to_public()

# ---------- WebSocket ----------
# C->S: {"type":"join","room_id":...,"name":...}
# C->S: {"type":"start","host_token":...}
# C->S: {"type":"guess","lat":...,"lon":...}
# S->C: {"type":"room_state", ...}
# S->C: {"type":"round_start","round_index":i,"name":str,"time_limit":sec}
# S->C: {"type":"guess_result","you":{...},"target":{"lat","lon"},"leaderboard":[...]}
# S->C: {"type":"game_over","leaderboard":[...]}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    player_id = uuid.uuid4().hex
    room: Optional[Room] = None
    player: Optional[Player] = None

    async def broadcast(payload: dict):
        if not room: return
        dead = []
        for pid, pl in list(room.players.items()):
            try:
                await pl.ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(pid)
        for pid in dead:
            room.players.pop(pid, None)

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")

            if t == "join":
                rid = msg.get("room_id"); name = (msg.get("name") or "Spelare")
                room = ROOMS.get(rid)
                if not room:
                    await ws.send_text(json.dumps({"type":"error","error":"Rum finns inte"})); continue
                player = Player(name, ws)
                room.players[player_id] = player
                await broadcast({"type":"room_state", **room.to_public()})

            elif t == "start":
                if not room: continue
                if msg.get("host_token") != room.host_token:
                    await ws.send_text(json.dumps({"type":"error","error":"Endast host får starta"})); continue
                if room.state != "lobby": continue
                room.state = "playing"
                room.round_index = 0
                room.current_target = room.targets[0]
                for p in room.players.values(): p.score_total, p.last_guess = 0, None
                await broadcast({"type":"round_start","round_index":0,"name":room.current_target["display_name"],"time_limit":room.time_limit})

            elif t == "guess":
                if not (room and player and room.state=="playing" and room.current_target): continue
                lat = float(msg.get("lat")); lon = float(msg.get("lon"))
                tgt = room.current_target
                dist = haversine_m(lat, lon, tgt["lat"], tgt["lon"])
                sc = score_from_distance_m(dist)
                player.last_guess = {"lat":lat,"lon":lon,"dist":dist,"score":sc}
                player.score_total += sc

                # Klara ronden när alla gissat
                if all(pl.last_guess is not None for pl in room.players.values()):
                    lb = sorted(
                        [{"name": pl.name, "score_total": pl.score_total} for pl in room.players.values()],
                        key=lambda x: -x["score_total"]
                    )
                    # Skicka resultat
                    for pl in room.players.values():
                        await pl.ws.send_text(json.dumps({
                            "type":"guess_result",
                            "you": pl.last_guess,
                            "target":{"lat":tgt["lat"], "lon":tgt["lon"]},
                            "leaderboard": lb
                        }))
                    # Reset per-rond
                    for pl in room.players.values(): pl.last_guess = None
                    # Nästa runda / game over
                    room.round_index += 1
                    if room.round_index >= room.rounds:
                        room.state = "ended"
                        await broadcast({"type":"game_over","leaderboard": lb})
                    else:
                        room.current_target = room.targets[room.round_index]
                        await broadcast({
                            "type":"round_start",
                            "round_index": room.round_index,
                            "name": room.current_target["display_name"],
                            "time_limit": room.time_limit
                        })

    except WebSocketDisconnect:
        if room and player_id in room.players:
            room.players.pop(player_id, None)
            if not room.players:
                ROOMS.pop(room.id, None)
