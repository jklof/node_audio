### Node Audio — Help and Quick Start

This guide shows the essential interactions to build and run graphs quickly: how to add nodes, connect them, manage processing, and the available keyboard/mouse shortcuts.

### Launching the app

- From the project root, run:

```bash
python main.py
```

On macOS you can also use `start.sh` to check/setup the environment and launch.

### Adding nodes

- Right‑click on empty canvas → pick a category → choose a node type.
- The node appears at your click position. Drag it by the header to move.

### Connecting nodes

- Left‑click an output socket (small circle on the right) and drag to an input socket (left) on another node.
- Connections must be from an output to an input. Incompatible types won’t connect.
- You can start dragging from the input as well; it will connect when dropped on a valid output.

### Editing nodes and connections

- Right‑click a node → Rename Node.
- Right‑click a node that can provide a clock (e.g., metronome) → Set as Clock Source. The current clock is shown on the toolbar.
- Right‑click a connection → Delete Connection.
- Select any node or connection and press Delete/Backspace to remove it.

### Processing controls

- Start processing: F5 (also via the Play button on the toolbar)
- Stop processing: F6 (also via the Stop button on the toolbar)
- View → Show Processing Load: toggles a small utilization bar on each node’s header while processing.

### View and navigation

- Pan: Middle‑mouse drag
- Zoom: Mouse wheel (zooms around the cursor)
- Fit all nodes in view: Ctrl+0 (Cmd+0 on macOS)
- Zoom in: Ctrl+= (Cmd+= on macOS)
- Zoom out: Ctrl+- (Cmd+- on macOS)

### File operations

- New/Clear graph: Ctrl+N (Cmd+N on macOS)
- Open graph: Ctrl+O (Cmd+O on macOS)
- Save graph: Ctrl+S (Cmd+S on macOS)
- Exit: Ctrl+Q (Cmd+Q on macOS)

Graphs are saved/loaded as JSON files.

### Tips

- Use the category menus on right‑click to discover available nodes (both core `plugins/` and optional `additional_plugins/`). Some features require extras from `additional_requirements.txt`.
- If you add a clock‑capable node and no clock is selected yet, it may become the clock automatically; you can change it via the node’s context menu later.
- Use Fit View (Ctrl/Cmd+0) frequently to recenter after building larger graphs.


