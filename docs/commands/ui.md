# yanex ui

Launch the interactive web interface for visual experiment management and exploration.

## Synopsis

```bash
yanex ui [--port PORT] [--host HOST]
```

## Description

The `ui` command starts a local web server that provides an interactive browser-based interface for managing and exploring experiments. The web UI offers visual experiment comparison, filtering, and detailed inspection capabilities.

## Options

### `--port PORT`
Port number for the web server (default: 8000)

### `--host HOST`
Host address to bind to (default: localhost)

## Examples

### Start Web UI

```bash
# Start on default port 8000
yanex ui

# Custom port
yanex ui --port 8080

# Make accessible on network
yanex ui --host 0.0.0.0 --port 8000
```

## Web UI Features

### Experiment Browser
- Visual list of all experiments with key metrics
- Sortable and filterable columns
- Quick status overview
- Search and filter capabilities

### Experiment Comparison
- Side-by-side comparison of multiple experiments
- Visual metric charts and graphs
- Parameter difference highlighting
- Easy identification of optimal configurations

### Experiment Details
- Complete metadata display
- Parameter and metric visualization
- Artifact inspection
- Git state and reproducibility information
- Log file viewing

### Filtering and Search
- Filter by status, tags, dates
- Search by name or description
- Quick access to recent experiments
- Saved filter configurations

## Accessing the Web UI

After starting the server, open your browser to:

```
http://localhost:8000
```

Or the custom port you specified:

```
http://localhost:8080  # If using --port 8080
```

The terminal will display the URL after the server starts:

```
Starting Yanex Web UI...
Server running at http://localhost:8000
Press Ctrl+C to stop the server
```

## Use Cases

### Visual Exploration
Use the web UI when you want to:
- Browse experiments visually
- Quickly compare multiple results
- Inspect artifacts and logs in a user-friendly interface
- Share experiment results with team members

### Interactive Analysis
The web UI is ideal for:
- Exploring hyperparameter sweep results
- Identifying optimal configurations
- Reviewing experiment metadata and parameters
- Inspecting training curves and metrics

### Team Collaboration
Share experiment results:
- Make UI accessible on network (use `--host 0.0.0.0`)
- Team members can browse and compare results
- Visual interface for non-technical stakeholders
- Easy sharing of successful configurations

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the web server:

```
^C
Shutting down Yanex Web UI...
Server stopped
```

## Troubleshooting

### Port Already in Use

If you see an error about the port being in use:

```bash
# Try a different port
yanex ui --port 8081
```

### Cannot Access in Browser

1. **Check the URL**: Ensure you're using the correct address shown in the terminal
2. **Firewall**: Check if your firewall is blocking the port
3. **Browser cache**: Try a different browser or incognito mode

### Server Won't Start

If the server fails to start:

1. **Check Python version**: Ensure you have Python 3.8+
2. **Check dependencies**: Web UI requires additional dependencies (automatically installed with yanex)
3. **Check logs**: Terminal output will show specific error messages

## Tips

- **Keep terminal open**: The web server runs in the foreground; keep the terminal open while using the UI
- **Multiple tabs**: You can open multiple browser tabs to the same UI instance
- **Bookmark**: Bookmark `http://localhost:8000` for quick access
- **Network access**: Only use `--host 0.0.0.0` on trusted networks

## Related Commands

- [`list`](list.md) - Command-line experiment listing
- [`compare`](compare.md) - Terminal-based comparison tool
- [`show`](show.md) - Command-line experiment details

## See Also

- [CLI Commands Overview](../cli-commands.md)
- [Best Practices](../best-practices.md)
