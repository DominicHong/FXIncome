import json
import datetime
import os
from typing import Optional, Any
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    """A single log entry for a specific date."""
    date: datetime.date
    capital: float = 0.0
    avg_ttm: float = 0.0
    tenor_positions: list[float] = field(default_factory=list)
    target_positions: list[float] = field(default_factory=list)
    delta_sizes: list[float] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    planned_trades: list[dict[str, Any]] = field(default_factory=list)
    executed_trades: list[dict[str, Any]] = field(default_factory=list)


class BacktestLogger:
    """A logger class to handle backtest log entries with date-based organization."""

    def __init__(self):
        """
        Initialize the BacktestLogger.
        """
        self.entries: dict[datetime.date, LogEntry] = {}  # date -> LogEntry

    def get_or_create_entry(self, date: datetime.date) -> LogEntry:
        """
        Get an existing log entry for the given date or create a new one.

        Args:
            date (datetime.date): The date for the log entry

        Returns:
            LogEntry: The log entry for the given date
        """
        if date not in self.entries:
            self.entries[date] = LogEntry(date=date)
        return self.entries[date]

    def log_position_update(
        self,
        date: datetime.date,
        capital: float,
        avg_ttm: float,
        tenor_positions: list[float],
        target_positions: list[float],
        delta_sizes: list[float],
        positions: list,
    ) -> None:
        """
        Log a position update for a specific date.

        Args:
            date (datetime.date): The date of the update
            capital (float): Current capital
            avg_ttm (float): Average time to maturity
            tenor_positions (list[float]): Current tenor positions
            target_positions (list[float]): Target positions
            delta_sizes (list[float]): Position changes
            positions (list[PositionCollection]): Current bond positions
        """
        entry = self.get_or_create_entry(date)

        # Round values
        entry.capital = round(capital, 0)
        entry.avg_ttm = round(avg_ttm, 2)
        entry.tenor_positions = [round(p, 0) for p in tenor_positions]
        entry.target_positions = [round(p, 0) for p in target_positions]
        entry.delta_sizes = [round(p, 0) for p in delta_sizes]
        entry.positions = positions.to_dict()
        
    def log_planned_trade(
        self,
        date: datetime.date,
        action: str,
        tenor: int,
        symbol: str,
        price: float,
        volume: float,
    ) -> None:
        """
        Log a planned trade for a specific date.

        Args:
            date (date): The date of the planned trade
            action (str): 'buy' or 'sell'
            tenor (int): The tenor index
            symbol (str): The bond symbol
            price (float): The planned trade price
            volume (float): The planned trade volume
        """
        entry = self.get_or_create_entry(date)
        entry.planned_trades.append({
            "action": action,
            "tenor": tenor,
            "symbol": symbol,
            "price": round(price, 4),
            "volume": round(volume, 0)
        })

    def log_executed_trade(
        self,
        date: datetime.date,
        direction: str,
        symbol: str,
        price: float,
        volume: float,
    ) -> None:
        """
        Log an executed trade for a specific date.

        Args:
            date (date): The date of the executed trade
            direction (str): The trade direction
            symbol (str): The bond symbol
            price (float): The executed trade price
            volume (float): The executed trade volume
        """
        entry = self.get_or_create_entry(date)
        entry.executed_trades.append({
            "direction": direction,
            "symbol": symbol,
            "price": round(price, 4),
            "volume": round(volume, 0)
        })

    
    def generate_json_report(self, output_path: str) -> None:
        """
        Generate a JSON report from the log entries.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Prepare entries for JSON serialization
        entries_list = list(self.entries.values())
        entries_list.sort(key=lambda x: x.date)

        def entry_to_dict(entry: LogEntry) -> dict:
            return {
                'date': entry.date.isoformat(),
                'capital': entry.capital,
                'avg_ttm': entry.avg_ttm,
                'tenor_positions': entry.tenor_positions,
                'target_positions': entry.target_positions,
                'delta_sizes': entry.delta_sizes,
                'positions': entry.positions,
                'planned_trades': entry.planned_trades,
                'executed_trades': entry.executed_trades,
            }

        data = [entry_to_dict(entry) for entry in entries_list]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_html_report(self, output_path: str) -> None:
        """
        Generate an HTML report from the log entries.

        Args:
            output_path (str): Path where the HTML report will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Get sorted entries
        entries_list = list(self.entries.values())
        entries_list.sort(key=lambda x: x.date)

        # Generate HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Index Enhancement Strategy Backtest Log Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; font-size: 13px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 12px; }
                th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .section { margin-bottom: 20px; }
                h1 { font-size: 20px; color: #333; }
                h2 { font-size: 16px; color: #333; margin: 0; }
                h3 { font-size: 14px; color: #333; margin: 10px 0 5px 0; }
                .trade { margin: 3px 0; padding: 4px; background-color: #f5f5f5; font-size: 12px; }
                .position { margin: 3px 0; font-size: 12px; }
                summary { cursor: pointer; padding: 5px; background-color: #f0f0f0; border: 1px solid #ddd; }
                summary:hover { background-color: #e0e0e0; }
                .date-header { display: flex; justify-content: space-between; align-items: center; }
                .date-header span { font-weight: bold; }
                .date-header .summary { font-size: 12px; color: #666; }
                button { font-size: 12px; padding: 4px 8px; }
            </style>
            <script>
                function toggleAll(expand) {
                    const details = document.querySelectorAll('details');
                    details.forEach(detail => {
                        detail.open = expand;
                    });
                }
            </script>
        </head>
        <body>
            <h1>Backtest Log Report</h1>
            <div style="margin-bottom: 10px;">
                <button onclick="toggleAll(true)" style="font-size: 11px; padding: 4px 8px;">Expand All</button>
                <button onclick="toggleAll(false)" style="font-size: 11px; padding: 4px 8px;">Collapse All</button>
            </div>
        """

        for entry in entries_list:
            # Calculate summary statistics
            total_positions = sum(pos['position'] for pos in entry.positions.get('positions', []))
            total_planned = len(entry.planned_trades)
            total_executed = len(entry.executed_trades)
            
            html_content += f"""
            <details class="section">
                <summary>
                    <div class="date-header">
                        <span>Date: {entry.date}</span>
                        <span class="summary">
                            Capital: {entry.capital:,.0f} | 
                            TTM: {entry.avg_ttm:.2f} | 
                            Positions: {total_positions:,.0f} | 
                            Planned: {total_planned} | 
                            Executed: {total_executed}
                        </span>
                    </div>
                </summary>
                <div style="padding: 10px;">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Capital</td>
                            <td>{entry.capital:,.2f}</td>
                        </tr>
                        <tr>
                            <td>Average TTM</td>
                            <td>{entry.avg_ttm:.2f}</td>
                        </tr>
                    </table>

                    <h3>Tenor Positions</h3>
                    <table>
                        <tr>
                            <th>Tenor</th>
                            <th>Current Position</th>
                            <th>Target Position</th>
                            <th>Delta Size</th>
                        </tr>
            """
            
            for i, (curr, target, delta) in enumerate(zip(entry.tenor_positions, entry.target_positions, entry.delta_sizes)):
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{curr:,.0f}</td>
                            <td>{target:,.0f}</td>
                            <td>{delta:,.0f}</td>
                        </tr>
                """

            html_content += """
                    </table>

                    <h3>Current Positions</h3>
                    <div class="positions">
            """
            
            for pos in entry.positions.get('positions', []):
                html_content += f"""
                        <div class="position">
                            {pos['symbol']}: {pos['position']:,.0f}
                        </div>
                """

            html_content += """
                    </div>

                    <h3>Planned Trades</h3>
                    <div class="trades">
            """
            
            for trade in entry.planned_trades:
                html_content += f"""
                        <div class="trade">
                            {trade['action'].upper()} {trade['symbol']} - 
                            Price: {trade['price']:.4f}, 
                            Volume: {trade['volume']:,.0f}
                        </div>
                """

            html_content += """
                    </div>

                    <h3>Executed Trades</h3>
                    <div class="trades">
            """
            
            for trade in entry.executed_trades:
                html_content += f"""
                        <div class="trade">
                            {trade['direction']} {trade['symbol']} - 
                            Price: {trade['price']:.4f}, 
                            Volume: {trade['volume']:,.0f}
                        </div>
                """

            html_content += """
                    </div>
                </div>
            </details>
            """

        html_content += """
        </body>
        </html>
        """

        # Write HTML content to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
