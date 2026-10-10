"""Alle dieren van C-01. Een nieuw dier = een nieuw bestand met een DIER-object, en hier toevoegen."""
from importlib import import_module

VOLGORDE = ["vos", "koe", "uil", "draak", "eenhoorn", "leeuw", "hert",
            # fan-art van bestaande personages: alleen voor eigen gebruik, niet publiceren of verkopen
            "mario", "sonic", "spiderman", "steve", "creeper"]

ALLE = [import_module(f"{__name__}.{naam}").DIER for naam in VOLGORDE]
EIGEN = [d for d in ALLE if not d.fanart]
FANART = [d for d in ALLE if d.fanart]
PER_SLEUTEL = {d.sleutel: d for d in ALLE}


def bouwbaar(dieren):
    """De dieren die hier gebouwd kunnen worden. Overgetrokken fan-art heeft zijn bronafbeelding nodig
    (fanart/bronnen/, niet in de publieke repo); ontbreekt die, dan slaan we hem over en blijven de
    bestaande bestanden staan."""
    out = []
    for d in dieren:
        try:
            d.tekening()
        except FileNotFoundError as e:
            print(f"  overgeslagen: {d.sleutel}: {e}")
            continue
        out.append(d)
    return out
