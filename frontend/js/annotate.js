// Annotorious integration: draw regions, save/load annotations.
//
// Attaches the Annotorious plugin to the OpenSeadragon viewer so the user can
// draw polygons/rectangles over the slide. Annotations are W3C annotation JSON;
// they are sent to the backend (via api.js) and stored in data/annotations/.
//
// This is Phase 2.
//
// TODO Phase 2:
//   - init Annotorious on the viewer created in viewer.js
//   - on create/update/delete, POST the annotation to the backend
//   - on load, fetch existing annotations and render them
