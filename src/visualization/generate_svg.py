"""
SVG Knowledge Graph Generator

Generates hierarchical SVG visualizations from extraction results JSON files.
Based on the example in docs/graph.svg
"""

import json
import sys
from typing import Dict, List, Tuple, Any
from pathlib import Path


# Color scheme for entity types
ENTITY_COLORS = {
    "fact": "#b2ebf2",        # cyan
    "hypothesis": "#ffe082",  # yellow
    "method": "#9fa8da",   # indigo
    "experiment": "#f48fb1",  # pink
    "result": "#a5d6a7",      # green
    "conclusion": "#b0bec5",  # grey
    "dataset": "#e1bee7",     # light purple
    "analysis": "#ffccbc"     # light orange
}

# Color scheme for relationship types
RELATIONSHIP_COLORS = {
    "fact_to_hypothesis": "#66BB6A",
    "hypothesis_to_method": "#FF9800",
    "hypothesis_to_experiment": "#FF9800",
    "method_to_result": "#9C27B0",
    "result_to_conclusion": "#1E88E5",
    "related_to": "#9E9E9E"
}

# Column configuration (X positions) - 8 separate columns
COLUMNS = {
    "fact": {"x": 150, "label": "Input Facts"},
    "hypothesis": {"x": 430, "label": "Hypotheses"},
    "experiment": {"x": 710, "label": "Experiments"},
    "method": {"x": 990, "label": "Techniques"},
    "result": {"x": 1270, "label": "Results"},
    "dataset": {"x": 1550, "label": "Datasets"},
    "analysis": {"x": 1830, "label": "Analysis"},
    "conclusion": {"x": 2110, "label": "Conclusions"}
}

# Layout parameters
NODE_WIDTH = 200
NODE_PADDING = 20
LINE_HEIGHT = 16
COLUMN_SPACING = 350
HEADER_HEIGHT = 50
LEGEND_HEIGHT = 100
MARGIN_TOP = 100


def escape_xml(text: str) -> str:
    """Escape special XML characters"""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def wrap_text(text: str, max_chars: int = 25) -> List[str]:
    """Wrap text into multiple lines"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + len(current_line) > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def calculate_node_height(text: str) -> int:
    """Calculate node height based on text length"""
    lines = wrap_text(text)
    return max(80, len(lines) * LINE_HEIGHT + NODE_PADDING * 2)


def generate_svg_markers() -> str:
    """Generate SVG marker definitions for arrows"""
    markers = []
    for rel_type, color in RELATIONSHIP_COLORS.items():
        markers.append(f'''        <marker id="arrow-{rel_type}" viewBox="0 -5 10 10" refX="8" refY="0"
                markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,-5L10,0L0,5" fill="{color}"/>
        </marker>''')

    return "\n".join(markers)


def generate_column_headers() -> str:
    """Generate column header text elements"""
    unique_columns = {}
    for entity_type, config in COLUMNS.items():
        x = config["x"]
        label = config["label"]
        if x not in unique_columns:
            unique_columns[x] = label

    headers = []
    for x, label in unique_columns.items():
        escaped_label = escape_xml(label)
        headers.append(f'''    <text x="{x}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#495057">
        {escaped_label}
    </text>''')

    return "\n".join(headers)


def generate_edge(source_pos: Tuple[float, float, float, float],
                  target_pos: Tuple[float, float, float, float],
                  rel_type: str) -> str:
    """Generate SVG path for edge with Bezier curve"""
    sx, sy, sw, sh = source_pos
    tx, ty, tw, th = target_pos

    # Start point: right edge of source node
    start_x = sx + sw
    start_y = sy + sh / 2

    # End point: left edge of target node
    end_x = tx
    end_y = ty + th / 2

    # Control points for Bezier curve
    ctrl1_x = start_x + (end_x - start_x) / 2
    ctrl1_y = start_y
    ctrl2_x = end_x - (end_x - start_x) / 2
    ctrl2_y = end_y

    color = RELATIONSHIP_COLORS.get(rel_type, RELATIONSHIP_COLORS["related_to"])
    stroke_dasharray = ' stroke-dasharray="6,3"' if rel_type == "fact_to_hypothesis" else ""

    path = f'''    <path d="M{start_x},{start_y} C{ctrl1_x},{ctrl1_y} {ctrl2_x},{ctrl2_y} {end_x},{end_y}"
          stroke="{color}" stroke-width="2" fill="none" opacity="0.8"
          marker-end="url(#arrow-{rel_type})"{stroke_dasharray}/>'''

    return path


def generate_node(entity: Dict[str, Any], x: float, y: float) -> Tuple[str, Tuple[float, float, float, float]]:
    """Generate SVG elements for a node"""
    text = entity.get("text", "")
    entity_type = entity.get("type", "fact")

    # Calculate node dimensions
    lines = wrap_text(text, max_chars=25)
    height = calculate_node_height(text)
    width = NODE_WIDTH

    # Adjust width for longer nodes
    if entity_type in ["method", "experiment", "hypothesis"]:
        width = min(NODE_WIDTH + 24, NODE_WIDTH + max(0, len(max(lines, key=len)) - 25) * 4)

    color = ENTITY_COLORS.get(entity_type, "#e0e0e0")

    # Generate rect
    rect = f'''    <rect x="{x - width/2}" y="{y}"
          width="{width}" height="{height}"
          fill="{color}" stroke="#495057" stroke-width="1.5" rx="6"/>'''

    # Generate text lines
    text_elements = []
    text_start_y = y + NODE_PADDING + LINE_HEIGHT / 2
    for i, line in enumerate(lines):
        line_y = text_start_y + i * LINE_HEIGHT
        escaped_line = escape_xml(line)
        text_elements.append(f'''    <text x="{x}" y="{line_y}" text-anchor="middle" dominant-baseline="middle"
          font-size="12" font-weight="500" fill="#212529">{escaped_line}</text>''')

    node_svg = rect + "\n" + "\n".join(text_elements)
    position = (x - width/2, y, width, height)

    return node_svg, position


def generate_legend() -> str:
    """Generate legend for relationship types"""
    legend_y = "LEGEND_Y_PLACEHOLDER"

    legend = [f'''    <text x="50" y="{legend_y}" font-size="14" font-weight="bold" fill="#495057">
        Legend:
    </text>''']

    rel_types = [
        ("fact_to_hypothesis", "Fact To Hypothesis"),
        ("hypothesis_to_method", "Hypothesis To Method"),
        ("method_to_result", "Method To Result"),
        ("result_to_conclusion", "Result To Conclusion")
    ]

    for i, (rel_type, label) in enumerate(rel_types):
        line_y = int(legend_y) + 20 + i * 30 if isinstance(legend_y, int) else f"{{LEGEND_Y + {20 + i * 30}}}"
        line_x1 = 50
        line_x2 = 90
        text_x = 100

        color = RELATIONSHIP_COLORS.get(rel_type, "#9E9E9E")
        dasharray = ' stroke-dasharray="6,3"' if rel_type == "fact_to_hypothesis" else ""
        escaped_label = escape_xml(label)

        legend.append(f'''    <line x1="{line_x1}" y1="{line_y}" x2="{line_x2}" y2="{line_y}"
          stroke="{color}" stroke-width="2"{dasharray}
          marker-end="url(#arrow-{rel_type})"/>
    <text x="{text_x}" y="{{LEGEND_Y + {24 + i * 30}}}" font-size="11" fill="#495057">
        {escaped_label}
    </text>''')

    return "\n".join(legend)


def generate_svg_from_json(result_json_path: str, output_svg_path: str = None) -> str:
    """
    Generate SVG from extraction result JSON

    Args:
        result_json_path: Path to the result JSON file
        output_svg_path: Path to save SVG (optional)

    Returns:
        SVG content as string
    """
    # Load JSON
    with open(result_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract entities and relationships
    entities_dict = data.get("entities", {})
    if isinstance(entities_dict, list):
        # Handle flat list format (graph.json)
        entities = {}
        for entity in entities_dict:
            entity_type = entity.get("type", "fact")
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity)
    else:
        # Handle grouped format (result.json)
        entities = entities_dict

    relationships = data.get("relationships", [])

    # Group entities by column
    columns_entities = {}
    for entity_type, entity_list in entities.items():
        col_x = COLUMNS.get(entity_type, {"x": 850})["x"]
        if col_x not in columns_entities:
            columns_entities[col_x] = []
        columns_entities[col_x].extend(entity_list)

    # Calculate layout
    entity_positions = {}
    max_y = MARGIN_TOP

    for col_x in sorted(columns_entities.keys()):
        col_entities = columns_entities[col_x]
        y = MARGIN_TOP

        for entity in col_entities:
            entity_id = entity.get("id", "")
            height = calculate_node_height(entity.get("text", ""))
            entity_positions[entity_id] = (entity, col_x, y, height)
            y += height + NODE_PADDING

        max_y = max(max_y, y)

    # Calculate SVG dimensions (8 columns layout)
    svg_width = 2400
    svg_height = max_y + LEGEND_HEIGHT + 50

    # Generate SVG
    svg_parts = []
    svg_parts.append(f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">')
    svg_parts.append('    <defs>')
    svg_parts.append(generate_svg_markers())
    svg_parts.append('    </defs>')
    svg_parts.append('    ')
    svg_parts.append(generate_column_headers())

    # Draw edges first (behind nodes)
    svg_parts.append('    <!-- Edges -->')
    for rel in relationships:
        source_id = rel.get("source_id", "")
        target_id = rel.get("target_id", "")
        rel_type = rel.get("relationship_type", "related_to")

        if source_id in entity_positions and target_id in entity_positions:
            source_entity, sx, sy, sh = entity_positions[source_id]
            target_entity, tx, ty, th = entity_positions[target_id]

            source_width = NODE_WIDTH
            if source_entity.get("type") in ["method", "experiment", "hypothesis"]:
                source_width = min(NODE_WIDTH + 24, NODE_WIDTH + 24)

            target_width = NODE_WIDTH
            if target_entity.get("type") in ["method", "experiment", "hypothesis"]:
                target_width = min(NODE_WIDTH + 24, NODE_WIDTH + 24)

            edge = generate_edge(
                (sx - source_width/2, sy, source_width, sh),
                (tx - target_width/2, ty, target_width, th),
                rel_type
            )
            svg_parts.append(edge)

    # Draw nodes
    svg_parts.append('    <!-- Nodes -->')
    for entity_id, (entity, x, y, height) in entity_positions.items():
        node_svg, _ = generate_node(entity, x, y)
        svg_parts.append(node_svg)

    # Draw legend
    legend_y = max_y + 30
    svg_parts.append('    <!-- Legend -->')
    legend_svg = generate_legend().replace("LEGEND_Y_PLACEHOLDER", str(legend_y))
    legend_svg = legend_svg.replace("{LEGEND_Y + 20}", str(legend_y + 20))
    legend_svg = legend_svg.replace("{LEGEND_Y + 24}", str(legend_y + 24))
    legend_svg = legend_svg.replace("{LEGEND_Y + 50}", str(legend_y + 50))
    legend_svg = legend_svg.replace("{LEGEND_Y + 54}", str(legend_y + 54))
    legend_svg = legend_svg.replace("{LEGEND_Y + 80}", str(legend_y + 80))
    legend_svg = legend_svg.replace("{LEGEND_Y + 84}", str(legend_y + 84))
    legend_svg = legend_svg.replace("{LEGEND_Y + 110}", str(legend_y + 110))
    legend_svg = legend_svg.replace("{LEGEND_Y + 114}", str(legend_y + 114))
    svg_parts.append(legend_svg)

    svg_parts.append('</svg>')

    svg_content = "\n".join(svg_parts)

    # Save to file if path provided
    if output_svg_path:
        with open(output_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"✅ SVG saved to: {output_svg_path}")

    return svg_content


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python generate_svg.py <result_json_path> [output_svg_path]")
        print("\nExample:")
        print("  python generate_svg.py results/sample_article_result.json")
        print("  python generate_svg.py results/sample_article_result.json results/graph.svg")
        sys.exit(1)

    result_json_path = sys.argv[1]

    # Auto-generate output path if not provided
    if len(sys.argv) >= 3:
        output_svg_path = sys.argv[2]
    else:
        # Generate output path based on input
        input_path = Path(result_json_path)
        output_svg_path = str(input_path.parent / f"{input_path.stem}_graph.svg")

    print(f"📊 Generating SVG from: {result_json_path}")

    try:
        generate_svg_from_json(result_json_path, output_svg_path)
        print(f"✨ Done!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
