# Flood Alert

Turns a broad flash flood warning into a place-by-place picture of which locations inside the warned area are most likely to flood, so residents know where the danger is rather than only that it exists.

## Language

### Places

**Intersection**:
A point where two or more named, drivable roads meet, identified by the pair of road names (with the road number where one exists). The only thing that receives a Susceptibility Rating.
_Avoid_: Junction, crossing, road, road segment, location

**Image Pair**:
Two aerial images of the same ground footprint: one taken before Hurricane Harvey and one taken in the days after it.
_Avoid_: Tile, before/after, sample

**Covered Area**:
The ground inside the footprint of at least one Image Pair. Only Intersections in the Covered Area can be rated.
_Avoid_: Study area, region, coverage

**Unrated**:
The status of an Intersection outside the Covered Area. It means nothing is known, which is different from Low.
_Avoid_: No risk, safe, unknown-low

### Ratings

**Observed Impact**:
What aerial imagery shows actually happened at a place during a past flood event. A historical fact about one event, not a prediction.
_Avoid_: Flood label, ground truth

**Susceptibility Rating**:
A standing judgement of how prone an Intersection is to flooding in heavy rain, taking one of four values: High, Moderate, Low or Undetermined. It is a property of the place and does not change from storm to storm.
_Avoid_: Risk score, flood risk, forecast, prediction

**High**:
The roadway at the Intersection is visibly under water in the after image and was not in the before image.

**Moderate**:
The roadway is not visibly submerged, but New Water or Flood Residue is visible close by, or the Intersection lies in a FEMA High-Risk Zone.

**Low**:
No New Water or Flood Residue is visible near the Intersection and it lies outside the FEMA High-Risk Zone. A weaker claim than High, because the after images were taken days after the heaviest rain.
_Avoid_: Safe, no risk

**Undetermined**:
The roadway cannot be seen in one or both images of the Image Pair, so no judgement is made. Different from Unrated, which means there is no imagery at all.
_Avoid_: Unknown, uncertain, unrated

### Labels

**Image Label**:
A yes / no / uncertain judgement about a whole image, recorded in the dataset itself: whether New Water is visible and whether Flood Residue is visible. It describes the image, not any Intersection inside it.
_Avoid_: Annotation, tag, rating

**Intersection Observation**:
What is seen at one Intersection across an Image Pair: whether the roadway is visible, whether New Water is on it, and whether New Water or Flood Residue is nearby. The Susceptibility Rating follows directly from it.
_Avoid_: Label, annotation, assessment

### Evidence

**New Water**:
Water present in the after image where there was none in the before image. Permanent rivers, lakes and ponds are not New Water.
_Avoid_: Flooding, inundation, standing water

**Flood Residue**:
Visible traces that water was recently present and has receded, such as mud, sediment or debris lines.
_Avoid_: Aftermath, damage

**FEMA High-Risk Zone**:
An area FEMA maps as a Special Flood Hazard Area, meaning a 1% or greater chance of flooding in any year. Where FEMA has no mapping, an Intersection is not known to be outside the zone and cannot be Low.
_Avoid_: Floodplain, flood zone, SFHA

**Close By**:
Within the patch of ground examined together with an Intersection, but not on its roadway.
_Avoid_: Near, adjacent, Nearby (which refers to distance from an Alert Location)

### Alerts

**Flash Flood Warning**:
An official National Weather Service warning covering a geographic area. It says where and when danger exists but not which places inside the area are worst.
_Avoid_: Alert

**Alert**:
A message for one Alert Location, produced when a Flash Flood Warning covers it, naming the Nearby Intersections rated High and Moderate. If the Alert Location is outside the Covered Area, it says the warning applies and that no ratings are available.
_Avoid_: Warning, notification, advisory

**Alert Location**:
A single point, supplied by a resident as an address or a map pin, that an Alert is written for.
_Avoid_: User location, home, subscriber

**Nearby**:
Within a fixed distance of the Alert Location. Only Nearby Intersections are named in an Alert.
_Avoid_: Local, in the area, close
