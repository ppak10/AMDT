import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from amdt.workspace import Workspace

console = Console()

class PortfolioWorkspace:
    """
    Portfolio class for Workspace methods
    """

    def create_workspace(self, name = None, portfolio_path = None, **kwargs):
        """
        Creates folder to store data related to AMDT workspace.

        @param name: Name of workspace
        @param portfolio_path: Override of portfolio path
        """

        # Sets `portfolio_path`` to value in self if override not provided.
        if portfolio_path is None:
            portfolio_path = self.portfolio_path
            
        workspace = Workspace(name=name)
        workspace_path = workspace.create_workspace(portfolio_path, **kwargs)

        # Print `create_workspace` success message.
        # print(textwrap.dedent(f"""
        # Workspace folder `{workspace.name}` at `{workspace_path}`.
        # Manage workspace with `manage.py` at `{workspace_path}`

        # ```
        # cd {workspace_path}
        # python manage.py
        # ```
        # """))
        console.print(
            textwrap.dedent(f"""
            [bold green]✅ Workspace created successfully![/bold green]

            [cyan]Workspace folder:[/cyan] [bold]{workspace.name}[/bold]
            [cyan]Location:[/cyan] [bold]{workspace_path}[/bold]

            Manage workspace with [magenta]manage.py[/magenta] at [underline]{workspace_path}[/underline]
            """),
            highlight=True
        )

        # Create a syntax-highlighted code block
        syntax = Syntax(f"cd {workspace_path}", "bash", theme="github-dark", line_numbers=False)

        # Wrap it in a bordered panel
        panel = Panel(
            syntax,  # Embed syntax highlighting inside the panel
            title="[cyan]Navigate to Workspace[/cyan]",  # Title on top
            border_style="blue",  # Blue border
            expand=False  # Keeps panel width minimal
        )
        console.print("Next Steps:")
        console.print(panel)

        return workspace
