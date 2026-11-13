# Grid Squares are US National Grid 1 km squares

Responders select Grid Squares and pass them to each other by Square Label, including over the radio and in shared links. We use the US National Grid's 1 km squares and their USNG references (e.g. "15R TU 52 07") as the grid and labels, not a grid of our own anchored to the project's data. USNG is the standard FEMA and many fire and search-and-rescue teams already use, so a square named from this map means the same ground on another agency's map.

## Considered Options

- **Project's own 1 km grid** (labels like "R12-C40"): simpler to compute and perfectly aligned to the web map, but labels mean nothing outside this app, and changing the anchor later would break every label already shared.
- **Squares that change size with zoom**: easiest to tap, but a square would mean different ground at different zooms, so it can't be handed off at all.

## Consequences

- Squares appear very slightly rotated against the web map, because they follow UTM, not latitude and longitude.
- The UTM zone 14/15 line at 96°W clips the squares it crosses. 22 Intersections (18 of them rated) lie west of it, around Simonton. Squares on either side carry their own zone's label.
