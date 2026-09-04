# ADR-0006: Ani-Mayhem Table Observer and Guide

**Status:** Accepted

**Date:** 2026-09-04

## Context

MultiVision already provides the physical-table primitives needed by Ani-Mayhem: calibrated ArUco detection, metric positions, spatial anchors, and realtime overlays. Ani-Mayhem already maps card marker IDs to game cards and can locate those cards on the calibrated table. Pyra Runtime already provides a small Unix-socket client for sending information into a Pi session.

The integration should remain deliberately small.

We do not want to recreate the original Ani-Mayhem board as a dense projected UI. In actual play, fixed outlines and dedicated projected slots for every possible game element would create visual clutter and make the physical table less flexible.

Instead, the physical table remains authoritative. Projection is used only to explain, highlight, point, and guide.

The desired interaction includes, for example:

- guiding setup step by step;
- highlighting cards that can currently be played;
- highlighting locations that can currently be scavenged;
- indicating which character is targeted in combat;
- showing which combat cards belong to which character;
- highlighting the cards participating in a damage calculation;
- showing where a card should be moved;
- displaying the current round / phase and a short instruction at the top of the projected surface;
- using a dedicated ArUco marker as a physical cursor when the user's intent cannot be inferred from the table state.

The integration must not introduce a second rules engine, coordinator service, event bus, or synthetic table simulator unless later experience demonstrates a concrete need.

## Decision

Implement exactly two Ani-Mayhem-facing integration components:

1. **Observer** — reads MultiVision tag detections and exposes stable, game-useful information about the physical table.
2. **Guide** — accepts game-useful presentation commands and translates them into MultiVision overlay operations.

Pyra / Ani-Mayhem game logic sits above these two components and decides what observations mean and what guidance to display.

```text
                 Pyra / Ani-Mayhem logic
                    ↑             ↓
                Observer         Guide
                    ↑             ↓
                       MultiVision
```

Observer and Guide share small data contracts, but no additional runtime component is introduced between them.

## Observer

### Responsibility

Observer converts MultiVision's current tag detections into a stable logical view of visible Ani-Mayhem objects.

It should expose both:

- the current snapshot of known visible objects; and
- semantic changes since the previous observation.

Observer is responsible for geometry and change detection only. It does not decide whether a move is legal, whether a phase is complete, or what should happen next in the game.

### Observed objects

At minimum an observed object contains:

- marker ID;
- resolved Ani-Mayhem card / token identity when known;
- metric table position;
- orientation when available;
- visibility state;
- optional pointer relation, e.g. the cursor currently indicating this object.

A small number of coarse regions may be added when they are genuinely useful, but the design does not depend on reconstructing the original board layout as a dense set of slots.

### Movement deadband

Observer uses the same movement assumptions as MultiVision:

- translation of **5 mm or less** is not reported as movement;
- rotation of **less than 10 degrees** is not reported as rotation.

Small changes are nevertheless retained as the latest object state.

This means the deadband is applied between consecutive retained states rather than against the last event-producing state. Slow physical drift therefore does not accumulate until it suddenly becomes a false movement event.

Example:

```text
100 mm -> 103 mm   no movement event; retained state = 103 mm
103 mm -> 107 mm   no movement event; retained state = 107 mm
107 mm -> 112 mm   no movement event; retained state = 112 mm
112 mm -> 140 mm   movement event
```

The same rule applies to orientation.

### Semantic events

The initial useful event set is intentionally small:

- object appeared;
- object disappeared;
- object moved beyond the deadband;
- object rotated beyond the deadband;
- cursor target changed.

Events may be batched when several physical changes are observed together.

Observer must not infer higher-level game events such as `CARD_PLAYED`, `SCAVENGE_STARTED`, or `COMBAT_RESOLVED`. Those interpretations belong to Ani-Mayhem / Pyra logic.

### Cursor

One dedicated ArUco marker is treated as a physical cursor.

The cursor is not a pixel-precise pointing device. Observer resolves it semantically to a nearby / overlapping object or projected choice target when possible.

Typical observations are therefore:

```text
cursor points to card #17
cursor points to choice "END TURN"
cursor points to nothing
```

The cursor is an escape hatch for ambiguous user intent, not a mandatory confirmation step after every physical action.

## Guide

### Responsibility

Guide converts a deliberately small set of semantic presentation commands into MultiVision overlay calls.

Guide does not know Ani-Mayhem rules and does not decide what should be highlighted.

### Initial command set

The initial command surface is:

- `highlight(objects, style)`
- `dim_except(objects)`
- `arrow(source, target)`
- `label(object, text)`
- `banner(text)`
- `top_bar(text, phase?, progress?)`
- `choice(options)`
- `clear()` / `clear_group(group)`

These primitives are sufficient for the currently identified interactions without encoding game-specific presentation logic inside MultiVision.

Examples:

- playable cards -> `highlight([...])`;
- available scavenging locations -> `highlight([...])`;
- combat target -> `highlight(character)`;
- combat card assignment -> `arrow(combat_card, character)` and/or `label(...)`;
- damage calculation -> `dim_except(participating_cards)`;
- required card movement -> `arrow(card, destination)`;
- phase display -> `top_bar("ROUND 3 · COMBAT")`.

### Top bar

Guide supports a persistent projected strip near the top of the usable surface.

It may contain:

- round number;
- current phase;
- setup / phase progress;
- one short instruction.

The top bar is UI chrome rather than part of the physical board model.

## Shared contracts

Observer and Guide should use explicit, small data structures rather than ad-hoc strings.

Expected contracts include equivalents of:

- `ObservedObject`;
- `BoardSnapshot`;
- `BoardEvent`;
- `GuideCommand`;
- `GuideTarget`.

Their exact Python representation is an implementation detail. They should remain serializable so that a thin CLI or socket adapter can be added without redesigning the domain objects.

## Pyra Runtime integration

Observer may send board updates to a specific Pyra Runtime Pi session using the existing Pyra Runtime message client and session-specific Unix socket.

Board observations are informational by default and should therefore use the existing default `wake=False` behavior.

The observer-to-Pyra transport is not part of MultiVision itself. MultiVision remains unaware of Ani-Mayhem and Pyra Runtime.

Guide likewise remains an Ani-Mayhem-side adapter over MultiVision rather than a game-specific feature inside MultiVision.

## Testing

No synthetic camera / fake-table renderer is required by this ADR.

Observer logic is tested using deterministic sequences of already-decoded MultiVision observations.

For example:

```text
frame 1: card #17 at (100, 100),   0°
frame 2: card #17 at (103, 101),   4°
frame 3: card #17 at (106, 103),   7°
frame 4: card #17 at (140, 103),   7°
```

Expected result:

```text
frames 1-3: no movement event
frame 4: movement event
```

Guide is tested by asserting that semantic Guide commands produce the expected MultiVision overlay operations.

End-to-end perception remains MultiVision's responsibility and is tested against the real camera/table pipeline rather than duplicated by Ani-Mayhem.

## Non-goals

This ADR explicitly does not introduce:

- a complete projected Ani-Mayhem board;
- dense permanent outlines for all possible card positions;
- an Ani-Mayhem rules engine inside MultiVision;
- a separate coordinator process between Observer and Guide;
- an event bus;
- a synthetic camera or fake-table input pipeline;
- mandatory cursor confirmation for every action;
- game-specific semantic inference inside Observer.

## Consequences

### Positive

- The integration remains small enough to reason about and test independently.
- MultiVision stays game-agnostic.
- Ani-Mayhem keeps ownership of game meaning and flow.
- Pyra receives concise information rather than raw camera data.
- Projected guidance can evolve without redesigning table geometry.
- Real tabletop layouts remain flexible and uncluttered.
- Cursor interaction is available where useful without becoming friction in normal play.

### Negative

- Higher-level game reasoning remains dependent on Ani-Mayhem / Pyra logic.
- Observer must maintain consistent object identities across detections.
- Cursor target resolution needs practical tuning on the real table.
- Some UI conventions (highlight styles, top-bar placement, choice layout) will need to be established empirically.

## Implementation order

1. Extract / reuse the existing Ani-Mayhem card-location logic as a library-friendly observer input path rather than repeatedly shelling out through CLI subprocesses.
2. Implement Observer snapshot + deadband + basic event diffing.
3. Implement cursor target resolution.
4. Implement Guide with `highlight`, `arrow`, `label`, `banner`, `top_bar`, and `clear` first; add `dim_except` and `choice` immediately when needed by the first guided flow.
5. Connect Observer messages to a selected Pyra Runtime session using its existing session socket.
6. Build the first vertical slice: guided Ani-Mayhem setup.
7. Extend the same primitives to round/action/combat guidance without expanding the architecture unless a concrete limitation appears.
