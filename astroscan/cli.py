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
    
    # Parse page numbers
    page_list = None
    if pages:
        page_list = _parse_pages(pages)
    
    # Single file mode
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
    
    # Batch mode
    from astroscan.pipeline import process_batch
    asyncio.run(process_batch(config, page_list, force, resume))


@cli.command()
@click.pass_context
def stats(ctx):
    """Show knowledge base statistics."""
    config = ctx.obj["config"]
    from astroscan.knowledge_base import KnowledgeBaseManager
    
    kb = KnowledgeBaseManager(config)
    s = kb.get_stats()
    
    click.echo("\n🧠 Knowledge Base Stats")
    click.echo("=" * 40)
    click.echo(f"  Total entries:      {s['total_entries']}")
    click.echo(f"  Total charts:       {s['total_charts']}")
    click.echo(f"  Pages processed:    {s['pages_processed']}")
    click.echo(f"  Definitions:        {s['total_definitions']}")
    click.echo(f"  Rules:              {s['total_rules']}")
    click.echo(f"  Signs indexed:      {s['signs_indexed']}")
    click.echo(f"  Planets indexed:    {s['planets_indexed']}")
    click.echo(f"  Houses indexed:     {s['houses_indexed']}")
    
    if s.get("categories"):
        click.echo(f"\n  Categories:")
        for cat, count in sorted(s["categories"].items()):
            click.echo(f"    {cat}: {count}")


@cli.command()
@click.argument("query")
@click.pass_context
def search(ctx, query):
    """Search the knowledge base."""
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


@cli.command("rebuild-kb")
@click.pass_context
def rebuild_kb(ctx):
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
        
        # Load metadata
        meta_path = page_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        page_num = meta.get("page_number", 0)
        
        # Add text
        text_path = page_dir / "text.md"
        if text_path.exists():
            kb.add_page_text(page_num, text_path.read_text())
        
        # Add knowledge entries
        ke_path = page_dir / "knowledge_entries.json"
        if ke_path.exists():
            added = kb.add_entries_from_json(ke_path.read_text(), page_num)
            count += added
        
        # Add charts
        chart_path = page_dir / "chart_analysis.md"
        if chart_path.exists():
            chart = ChartDescription(
                page_number=page_num,
                description=chart_path.read_text(),
                raw_analysis=chart_path.read_text(),
            )
            kb.add_chart(chart)
    
    kb.save()
    click.echo(f"\n✅ Knowledge base rebuilt: {count} entries from {kb.kb.total_pages_processed} pages")


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
        Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        out_path = output or "knowledge_base_export.md"
        lines = [f"# {kb.kb.source_name} — Knowledge Base\n"]
        lines.append(f"Pages processed: {kb.kb.total_pages_processed}\n")
        lines.append(f"Total entries: {len(kb.kb.entries)}\n\n---\n")
        
        for entry in kb.kb.entries:
            lines.append(f"## [{entry.category.value}] {entry.title} (p.{entry.page_number})\n")
            lines.append(f"{entry.content}\n")
            if entry.tags:
                lines.append(f"*Tags: {', '.join(entry.tags)}*\n")
            lines.append("\n---\n")
        
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    
    click.echo(f"✅ Exported to {out_path}")


def _parse_pages(pages_str: str) -> list[int]:
    """Parse page specification like '1-50' or '42,56,78'."""
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
