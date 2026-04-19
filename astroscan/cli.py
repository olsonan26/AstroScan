"""CLI interface for AstroScan."""
from __future__ import annotations
import asyncio
import click
from pathlib import Path

from astroscan.config import Config


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """🔭 AstroScan — Book-to-knowledge-base pipeline for astrology textbooks."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(config)


@cli.command()
@click.option("--file", "-f", help="Process a specific image file")
@click.option("--pages", "-p", help="Page numbers to process (e.g. '1-50' or '42,56,78')")
@click.option("--force", is_flag=True, help="Reprocess already-completed pages")
@click.option("--resume/--no-resume", default=True, help="Skip already-processed pages")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def process(ctx, file, pages, force, resume, verbose):
    """Process book page images through the full pipeline."""
    config = ctx.obj["config"]

    if not config.openrouter_api_key or config.openrouter_api_key == "sk-or-v1-your-key-here":
        click.echo("❌ Please set your OpenRouter API key in config.yaml!")
        click.echo("   Get a free key at: https://openrouter.ai/keys")
        return

    page_list = None
    if pages:
        page_list = _parse_pages(pages)

    if file:
        from astroscan.pipeline import process_single_page
        from astroscan.vision import VisionAI
        from astroscan.knowledge_base import KnowledgeBaseManager
        from astroscan.pipeline import _extract_page_number

        config.ensure_dirs()
        image_path = Path(file)
        if not image_path.exists():
            click.echo(f"❌ File not found: {file}")
            return

        page_num = _extract_page_number(image_path.stem)
        if page_num == 0:
            page_num = 1

        vision_ai = VisionAI(config)
        kb_manager = KnowledgeBaseManager(config)
        asyncio.run(process_single_page(
            image_path, page_num, config, vision_ai, kb_manager, force
        ))
        return

    from astroscan.pipeline import process_batch
    asyncio.run(process_batch(config, page_list, force, resume))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics (includes vector + graph stats)."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    s = kb.get_stats()

    click.echo("\n🧠 Knowledge Base Stats")
    click.echo("=" * 50)
    click.echo(f"  Total entries:      {s['total_entries']}")
    click.echo(f"  Total charts:       {s['total_charts']}")
    click.echo(f"  Pages processed:    {s['pages_processed']}")
    click.echo(f"  Definitions:        {s['total_definitions']}")
    click.echo(f"  Rules:              {s['total_rules']}")
    click.echo(f"  Signs indexed:      {s['signs_indexed']}")
    click.echo(f"  Planets indexed:    {s['planets_indexed']}")
    click.echo(f"  Houses indexed:     {s['houses_indexed']}")

    if s.get("categories"):
        click.echo(f"\n  📂 Categories:")
        for cat, count in sorted(s["categories"].items()):
            click.echo(f"    {cat}: {count}")

    # ChromaDB stats
    click.echo(f"\n  🔍 ChromaDB Vectors: {s.get('vector_count', 0)}")

    # Graph stats
    click.echo(f"\n  🕸️  Knowledge Graph:")
    click.echo(f"    Nodes: {s.get('graph_nodes', 0)}")
    click.echo(f"    Edges: {s.get('graph_edges', 0)}")

    node_types = s.get("graph_node_types", {})
    if node_types:
        click.echo(f"    Node types:")
        for t, c in sorted(node_types.items()):
            click.echo(f"      {t}: {c}")

    top = s.get("top_connected", [])
    if top:
        click.echo(f"\n  🌟 Most Connected Concepts:")
        for item in top[:10]:
            click.echo(f"    {item['name']}: {item['connections']} connections")


@cli.command()
@click.argument("query")
@click.pass_context
def search(ctx, query):
    """Keyword search the knowledge base (legacy)."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    results = kb.search(query)

    if not results:
        click.echo(f"No results for '{query}'")
        return

    click.echo(f"\n🔍 Results for '{query}' ({len(results)} found):\n")
    for entry in results[:10]:
        click.echo(f"  [{entry.category.value.upper()}] {entry.title} (p.{entry.page_number})")
        click.echo(f"    {entry.content[:120]}...")
        if entry.tags:
            click.echo(f"    Tags: {', '.join(entry.tags)}")
        click.echo()


@cli.command("semantic-search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--category", "-c", default=None, help="Filter by category")
@click.pass_context
def semantic_search(ctx, query, limit, category):
    """Search by MEANING using ChromaDB vectors (AI-powered)."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    results = kb.semantic_search(query, n_results=limit, category=category)

    if not results:
        click.echo(f"No semantic results for '{query}'")
        return

    click.echo(f"\n🧠 Semantic results for '{query}' ({len(results)} found):\n")
    for r in results:
        meta = r.get("metadata", {})
        dist = r.get("distance")
        dist_str = f" (distance: {dist:.4f})" if dist is not None else ""
        click.echo(f"  [{meta.get('category', '?').upper()}] {meta.get('title', r['id'])}{dist_str}")
        content = r.get("content", "")
        click.echo(f"    {content[:150]}...")
        click.echo()


@cli.command("graph-search")
@click.argument("concept")
@click.option("--depth", "-d", default=2, help="Traversal depth")
@click.pass_context
def graph_search_cmd(ctx, concept, depth):
    """Traverse the knowledge graph from a concept."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    result = kb.graph_search(concept, depth=depth)

    if not result["found"]:
        click.echo(f"Concept '{concept}' not found in knowledge graph.")
        click.echo(f"Available nodes: {', '.join(sorted(list(kb.graph.nodes)[:20]))}...")
        return

    click.echo(f"\n🕸️  Graph search: {concept}")
    click.echo(f"  Type: {result['node_type']}")
    click.echo(f"  Referenced in {len(result['entry_ids'])} entries")
    click.echo(f"\n  Connections ({len(result['connections'])}):")
    for conn in result["connections"]:
        arrow = "→" if conn["depth"] == 1 else "  →"
        click.echo(f"    {arrow} {conn['from']} --[{conn['relationship']}]--> {conn['to']}")


@cli.command("find-path")
@click.argument("concept_a")
@click.argument("concept_b")
@click.pass_context
def find_path(ctx, concept_a, concept_b):
    """Find connection paths between two concepts."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    paths = kb.find_connections(concept_a, concept_b)

    if not paths:
        click.echo(f"No path found between '{concept_a}' and '{concept_b}'.")
        return

    click.echo(f"\n🔗 Paths from {concept_a} → {concept_b}:\n")
    for i, path in enumerate(paths, 1):
        click.echo(f"  Path {i}: {' → '.join(path)}")


@cli.command("communities")
@click.pass_context
def communities(ctx):
    """Show concept clusters/communities in the knowledge graph."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    comms = kb.get_communities()

    if not comms:
        click.echo("No communities found. Process more pages first.")
        return

    click.echo(f"\n🏘️  Knowledge Communities ({len(comms)} clusters):\n")
    for cid, members in sorted(comms.items()):
        click.echo(f"  Cluster {cid} ({len(members)} concepts):")
        click.echo(f"    {', '.join(members[:15])}")
        if len(members) > 15:
            click.echo(f"    ... and {len(members) - 15} more")
        click.echo()


@cli.command("hybrid-search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.pass_context
def hybrid_search_cmd(ctx, query, limit):
    """🔥 BEST search — combines semantic + graph + keyword."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager

    kb = KnowledgeBaseManager(config)
    result = kb.hybrid_search(query, n_results=limit)

    click.echo(f"\n⚡ Hybrid search: '{query}'")
    click.echo(f"  Semantic matches: {result['semantic_count']}")
    click.echo(f"  Graph concepts found: {result['graph_concepts_found']}")
    click.echo(f"  Keyword matches: {result['keyword_count']}")
    click.echo(f"  Total results: {result['total_results']}\n")

    for r in result["results"]:
        source_emoji = {"semantic": "🧠", "graph": "🕸️", "keyword": "🔍"}.get(
            r.get("source", ""), "📄"
        )
        meta = r.get("metadata", {})
        click.echo(f"  {source_emoji} [{meta.get('category', '?').upper()}] {r['id']}")
        content = r.get("content", "")[:150]
        click.echo(f"    {content}...")
        click.echo()

    if result["graph_connections"]:
        click.echo("  🕸️  Related graph connections:")
        for gc in result["graph_connections"]:
            click.echo(f"    {gc['concept']} ({gc['node_type']}): "
                       f"{len(gc['connections'])} connections")


@cli.command("rebuild-kb")
@click.option("--vectors/--no-vectors", default=True, help="Also rebuild ChromaDB + graph")
@click.pass_context
def rebuild_kb(ctx, vectors):
    """Rebuild knowledge base from existing output files."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager
    from astroscan.models import ChartDescription
    import json

    kb = KnowledgeBaseManager(config)
    kb.kb.entries.clear()
    kb.kb.charts.clear()

    output_dir = config.output_dir
    if not output_dir.exists():
        click.echo("❌ No output directory found. Run 'process' first.")
        return

    count = 0
    for page_dir in sorted(output_dir.iterdir()):
        if not page_dir.is_dir() or not page_dir.name.startswith("page_"):
            continue

        meta_path = page_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        page_num = meta.get("page_number", 0)

        text_path = page_dir / "text.md"
        if text_path.exists():
            kb.add_page_text(page_num, text_path.read_text())

        ke_path = page_dir / "knowledge_entries.json"
        if ke_path.exists():
            added = kb.add_entries_from_json(ke_path.read_text(), page_num)
            count += added

        chart_path = page_dir / "chart_analysis.md"
        if chart_path.exists():
            chart = ChartDescription(
                page_number=page_num,
                description=chart_path.read_text(),
                raw_analysis=chart_path.read_text(),
            )
            kb.add_chart(chart)

    kb.save()

    if vectors:
        click.echo("\n  Rebuilding vector store + knowledge graph...")
        kb.rebuild_vectors_and_graph()

    click.echo(f"\n✅ Knowledge base rebuilt: {count} entries from "
               f"{kb.kb.total_pages_processed} pages")
    s = kb.get_stats()
    click.echo(f"   Vectors: {s.get('vector_count', 0)} | "
               f"Graph: {s.get('graph_nodes', 0)} nodes, "
               f"{s.get('graph_edges', 0)} edges")


@cli.command()
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "markdown"]), default="json")
@click.option("--output", "-o", default=None, help="Output file path")
@click.pass_context
def export(ctx, fmt, output):
    """Export knowledge base to a file."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager
    import json

    kb = KnowledgeBaseManager(config)

    if fmt == "json":
        out_path = output or "knowledge_base_export.json"
        data = kb.kb.model_dump()
        # Include graph stats
        data["_graph_stats"] = {
            "nodes": kb.graph.number_of_nodes(),
            "edges": kb.graph.number_of_edges(),
        }
        Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        out_path = output or "knowledge_base_export.md"
        lines = [f"# {kb.kb.source_name} — Knowledge Base\n"]
        lines.append(f"Pages processed: {kb.kb.total_pages_processed}\n")
        lines.append(f"Total entries: {len(kb.kb.entries)}\n")
        lines.append(f"Graph: {kb.graph.number_of_nodes()} nodes, "
                     f"{kb.graph.number_of_edges()} edges\n\n---\n")

        for entry in kb.kb.entries:
            lines.append(f"## [{entry.category.value}] {entry.title} (p.{entry.page_number})\n")
            lines.append(f"{entry.content}\n")
            if entry.tags:
                lines.append(f"*Tags: {', '.join(entry.tags)}*\n")
            lines.append("\n---\n")

        Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    click.echo(f"✅ Exported to {out_path}")


@cli.command("check-page")
@click.argument("image_path")
@click.pass_context
def check_page(ctx, image_path):
    """Analyze a page image before processing (curvature, quality, etc)."""
    from astroscan.dewarper import estimate_curvature
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        click.echo(f"❌ File not found: {image_path}")
        return

    img = Image.open(path)
    w, h = img.size
    size_mb = path.stat().st_size / (1024 * 1024)

    click.echo(f"\n📄 Page Analysis: {path.name}")
    click.echo(f"  Dimensions: {w}×{h}")
    click.echo(f"  File size:  {size_mb:.1f} MB")

    curvature = estimate_curvature(path)
    click.echo(f"\n📐 Curvature Analysis:")
    click.echo(f"  Score:          {curvature['curvature_score']:.2f} (0=flat, 1=very curved)")
    click.echo(f"  Needs dewarping: {'Yes' if curvature['needs_dewarping'] else 'No'}")
    if curvature['needs_dewarping']:
        click.echo(f"  Recommended:    {curvature['estimated_method']} method")


@cli.command("engines")
@click.pass_context
def engines(ctx):
    """Show which OCR engines are available."""
    click.echo("\n⚙️  OCR Engine Status:\n")

    # MinerU
    from astroscan.mineru_ocr import is_mineru_available
    if is_mineru_available():
        click.echo("  ✅ MinerU — INSTALLED (33k⭐, beats GPT-4o)")
    else:
        click.echo("  ❌ MinerU — not installed")
        click.echo("     Install: pip install magic-pdf[full]")

    # Marker
    try:
        from marker.converters.pdf import PdfConverter  # noqa: F401
        click.echo("  ✅ Marker — INSTALLED (23k⭐)")
    except ImportError:
        click.echo("  ❌ Marker — not installed")
        click.echo("     Install: pip install marker-pdf")

    # Surya
    try:
        from surya.recognition import RecognitionPredictor  # noqa: F401
        click.echo("  ✅ Surya — INSTALLED (14k⭐)")
    except ImportError:
        click.echo("  ❌ Surya — not installed (bundled with Marker)")

    # DocScanner model
    model_path = Path(__file__).parent.parent / "models" / "docscanner.pth"
    if model_path.exists():
        click.echo("  ✅ DocScanner model — AVAILABLE")
    else:
        click.echo("  ℹ️  DocScanner model — not downloaded (geometric dewarping still works)")

    # ChromaDB
    try:
        import chromadb  # noqa: F401
        click.echo("  ✅ ChromaDB — INSTALLED (semantic search)")
    except ImportError:
        click.echo("  ❌ ChromaDB — not installed (pip install chromadb)")

    # NetworkX
    try:
        import networkx  # noqa: F401
        click.echo("  ✅ NetworkX — INSTALLED (knowledge graph)")
    except ImportError:
        click.echo("  ❌ NetworkX — not installed (pip install networkx)")

    click.echo("\n  💡 Vision AI OCR is always available as fallback (free models via OpenRouter)")


@cli.command("dewarp")
@click.argument("image_path")
@click.option("--output", "-o", default=None, help="Output path (default: same dir, _dewarped suffix)")
@click.option("--method", "-m", type=click.Choice(["auto", "docscanner", "geometric"]),
              default="auto", help="Dewarping method")
@click.pass_context
def dewarp_cmd(ctx, image_path, output, method):
    """Dewarp a single curved/warped book page photo."""
    from astroscan.dewarper import dewarp_page

    path = Path(image_path)
    if not path.exists():
        click.echo(f"❌ File not found: {image_path}")
        return

    if output is None:
        output = str(path.parent / f"{path.stem}_dewarped{path.suffix}")

    try:
        result = dewarp_page(path, output, method=method)
        click.echo(f"✅ Dewarped: {result}")
    except Exception as e:
        click.echo(f"❌ Dewarping failed: {e}")


def _parse_pages(pages_str: str) -> list[int]:
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


if __name__ == "__main__":
    cli()
