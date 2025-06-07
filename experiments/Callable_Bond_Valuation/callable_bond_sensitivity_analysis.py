import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import logging

# Import FinancePy classes
from financepy.products.bonds.bond import Bond
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.day_count import DayCountTypes

from experiments.Callable_Bond_Valuation.interest_rate_simulator import (
    InterestRateSimulator,
)
from experiments.Callable_Bond_Valuation.callable_bond_valuer import (
    CallableBond,
    CallableBondValuer,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CallableBondSensitivityAnalyzer:
    """Comprehensive sensitivity analysis for callable bond valuation."""
    
    def __init__(
        self,
        base_issue_date: Date,
        base_maturity_date: Date,
        base_coupon_rate: float,
        base_straight_bond_ytm: float,
        base_sigma: float,
        call_protection_years: int = 1,
        face_value: float = 100.0,
        freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
        day_count_type: DayCountTypes = DayCountTypes.THIRTY_E_360,
        num_simulations: int = 2000,
        call_premium: float = 0.0,
    ):
        """Initialize the sensitivity analyzer with base parameters.
        
        Args:
            base_issue_date: Base bond issue date
            base_maturity_date: Base bond maturity date  
            base_coupon_rate: Base annual coupon rate
            base_straight_bond_ytm: Base straight bond YTM
            base_sigma: Base volatility for rate simulator
            call_protection_years: Years before bond becomes callable
            face_value: Face value of the bond
            freq_type: Coupon payment frequency
            day_count_type: Day count convention
            num_simulations: Number of Monte Carlo simulations
            call_premium: Premium above par for calling
        """
        self.base_issue_date = base_issue_date
        self.base_maturity_date = base_maturity_date
        self.base_coupon_rate = base_coupon_rate
        self.base_straight_bond_ytm = base_straight_bond_ytm
        self.base_sigma = base_sigma
        self.call_protection_years = call_protection_years
        self.face_value = face_value
        self.freq_type = freq_type
        self.day_count_type = day_count_type
        self.num_simulations = num_simulations
        self.call_premium = call_premium
        
        # Calculate base time to maturity in years
        self.base_time_to_maturity = (base_maturity_date - base_issue_date) / 365.25
        
        # Store results
        self.sensitivity_results: dict[str, pd.DataFrame] = {}
        
    def _create_callable_bond(
        self, 
        issue_date: Date, 
        maturity_date: Date, 
        coupon_rate: float
    ) -> CallableBond:
        """Create a callable bond with specified parameters."""
        return CallableBond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            call_protection_years=self.call_protection_years,
            face_value=self.face_value,
            freq_type=self.freq_type,
            day_count_type=self.day_count_type,
        )
    
    def _create_rate_simulator(
        self, 
        time_to_maturity_years: float, 
        sigma: float,
        r0: float = 0.0
    ) -> InterestRateSimulator:
        """Create an interest rate simulator with specified parameters."""
        days = int(time_to_maturity_years * 365.25) + 2
        days_of_year = 365
        
        return InterestRateSimulator(
            r0=r0,
            mu=0,
            sigma=sigma,
            days=days,
            days_of_year=days_of_year,
            num_paths=self.num_simulations,
            model="abm",
            kappa=0.1,
        )
    
    def _value_callable_bond(
        self,
        coupon_rate: float,
        time_to_maturity_years: float,
        sigma: float,
        straight_bond_ytm: float,
    ) -> tuple[float, float, dict[str, Any]]:
        """Value a callable bond with specified parameters using the same approach as solve_callable_bond_coupon().
        
        Returns:
            Tuple of (callable_bond_value, straight_bond_value, additional_stats)
        """
        # Create dates based on time to maturity
        issue_date = self.base_issue_date
        maturity_date = issue_date.add_years(int(time_to_maturity_years))
        
        # Create callable bond
        callable_bond = self._create_callable_bond(issue_date, maturity_date, coupon_rate)
        
        # Calculate equivalent initial short rate
        equivalent_rate, _ = CallableBondValuer.calculate_equivalent_initial_short_rate(
            straight_bond=callable_bond.bond,
            dt=1/365,  # Daily time step
            ytm=straight_bond_ytm,
            valuation_date=issue_date,
            discount_mode="discrete",
            tolerance=1e-6,
        )
        
        # Create rate simulator with equivalent rate
        rate_simulator = self._create_rate_simulator(time_to_maturity_years, sigma, equivalent_rate)
        
        # Create valuer and value the bond
        valuer = CallableBondValuer(callable_bond, rate_simulator, self.call_premium)
        
        callable_value, straight_value, stats = valuer.value_bond(
            valuation_date=issue_date,
            straight_bond_ytm=straight_bond_ytm,
            num_simulations=self.num_simulations,
        )
        
        return callable_value, straight_value, stats
    
    def analyze_coupon_sensitivity(
        self, 
        coupon_range: list[float] = None
    ) -> pd.DataFrame:
        """Analyze sensitivity to coupon rate changes.
        
        Args:
            coupon_range: List of coupon rates to test
            
        Returns:
            DataFrame with sensitivity results
        """
        if coupon_range is None:
            coupon_range = np.arange(0.02, 0.08, 0.005).tolist()  # 2% to 8% in 0.5% steps
            
        logger.info(f"Analyzing coupon sensitivity with {len(coupon_range)} values...")
        
        results = []
        
        for coupon in coupon_range:
            try:
                callable_value, straight_value, stats = self._value_callable_bond(
                    coupon_rate=coupon,
                    time_to_maturity_years=self.base_time_to_maturity,
                    sigma=self.base_sigma,
                    straight_bond_ytm=self.base_straight_bond_ytm,
                )
                
                results.append({
                    'coupon_rate': coupon,
                    'callable_bond_value': callable_value,
                    'straight_bond_value': straight_value,
                    'option_value': stats['option_value'],
                    'call_probability': stats['call_probability'],
                    'callable_values_std': stats['callable_values_std'],
                })
                
            except Exception as e:
                logger.warning(f"Failed to value bond with coupon {coupon:.2%}: {e}")
                continue
                
        sensitivity_df = pd.DataFrame(results)
        self.sensitivity_results['coupon'] = sensitivity_df
        
        logger.info(f"Completed coupon sensitivity analysis with {len(sensitivity_df)} successful valuations")
        return sensitivity_df
    
    def analyze_maturity_sensitivity(
        self, 
        maturity_years_range: list[float] = None
    ) -> pd.DataFrame:
        """Analyze sensitivity to time to maturity changes.
        
        Args:
            maturity_years_range: List of time to maturity values (in years) to test
            
        Returns:
            DataFrame with sensitivity results
        """
        if maturity_years_range is None:
            maturity_years_range = np.arange(2, 12, 1).tolist()  # 2 to 11 years
            
        logger.info(f"Analyzing maturity sensitivity with {len(maturity_years_range)} values...")
        
        results = []
        
        for maturity_years in maturity_years_range:
            try:
                callable_value, straight_value, stats = self._value_callable_bond(
                    coupon_rate=self.base_coupon_rate,
                    time_to_maturity_years=maturity_years,
                    sigma=self.base_sigma,
                    straight_bond_ytm=self.base_straight_bond_ytm,
                )
                
                results.append({
                    'time_to_maturity_years': maturity_years,
                    'callable_bond_value': callable_value,
                    'straight_bond_value': straight_value,
                    'option_value': stats['option_value'],
                    'call_probability': stats['call_probability'],
                    'callable_values_std': stats['callable_values_std'],
                })
                
            except Exception as e:
                logger.warning(f"Failed to value bond with maturity {maturity_years} years: {e}")
                continue
                
        sensitivity_df = pd.DataFrame(results)
        self.sensitivity_results['maturity'] = sensitivity_df
        
        logger.info(f"Completed maturity sensitivity analysis with {len(sensitivity_df)} successful valuations")
        return sensitivity_df
    
    def analyze_sigma_sensitivity(
        self, 
        sigma_range: list[float] = None
    ) -> pd.DataFrame:
        """Analyze sensitivity to volatility (sigma) changes.
        
        Args:
            sigma_range: List of sigma values to test
            
        Returns:
            DataFrame with sensitivity results
        """
        if sigma_range is None:
            sigma_range = np.arange(0.005, 0.025, 0.0025).tolist()  # 0.5% to 2.5% in 0.25% steps
            
        logger.info(f"Analyzing sigma sensitivity with {len(sigma_range)} values...")
        
        results = []
        
        for sigma in sigma_range:
            try:
                callable_value, straight_value, stats = self._value_callable_bond(
                    coupon_rate=self.base_coupon_rate,
                    time_to_maturity_years=self.base_time_to_maturity,
                    sigma=sigma,
                    straight_bond_ytm=self.base_straight_bond_ytm,
                )
                
                results.append({
                    'sigma': sigma,
                    'callable_bond_value': callable_value,
                    'straight_bond_value': straight_value,
                    'option_value': stats['option_value'],
                    'call_probability': stats['call_probability'],
                    'callable_values_std': stats['callable_values_std'],
                })
                
            except Exception as e:
                logger.warning(f"Failed to value bond with sigma {sigma:.3f}: {e}")
                continue
                
        sensitivity_df = pd.DataFrame(results)
        self.sensitivity_results['sigma'] = sensitivity_df
        
        logger.info(f"Completed sigma sensitivity analysis with {len(sensitivity_df)} successful valuations")
        return sensitivity_df
    
    def analyze_ytm_sensitivity(
        self, 
        ytm_range: list[float] = None
    ) -> pd.DataFrame:
        """Analyze sensitivity to straight bond YTM changes.
        
        Args:
            ytm_range: List of YTM values to test
            
        Returns:
            DataFrame with sensitivity results
        """
        if ytm_range is None:
            ytm_range = np.arange(0.02, 0.08, 0.005).tolist()  # 2% to 8% in 0.5% steps
            
        logger.info(f"Analyzing YTM sensitivity with {len(ytm_range)} values...")
        
        results = []
        
        for ytm in ytm_range:
            try:
                callable_value, straight_value, stats = self._value_callable_bond(
                    coupon_rate=self.base_coupon_rate,
                    time_to_maturity_years=self.base_time_to_maturity,
                    sigma=self.base_sigma,
                    straight_bond_ytm=ytm,
                )
                
                results.append({
                    'straight_bond_ytm': ytm,
                    'callable_bond_value': callable_value,
                    'straight_bond_value': straight_value,
                    'option_value': stats['option_value'],
                    'call_probability': stats['call_probability'],
                    'callable_values_std': stats['callable_values_std'],
                })
                
            except Exception as e:
                logger.warning(f"Failed to value bond with YTM {ytm:.2%}: {e}")
                continue
                
        sensitivity_df = pd.DataFrame(results)
        self.sensitivity_results['ytm'] = sensitivity_df
        
        logger.info(f"Completed YTM sensitivity analysis with {len(sensitivity_df)} successful valuations")
        return sensitivity_df
    
    def run_full_sensitivity_analysis(
        self,
        coupon_range: list[float] = None,
        maturity_years_range: list[float] = None,
        sigma_range: list[float] = None,
        ytm_range: list[float] = None,
    ) -> dict[str, pd.DataFrame]:
        """Run complete sensitivity analysis for all parameters.
        
        Returns:
            Dictionary of DataFrames with sensitivity results for each parameter
        """
        logger.info("Starting comprehensive callable bond sensitivity analysis...")
        
        # Run individual sensitivity analyses
        self.analyze_coupon_sensitivity(coupon_range)
        self.analyze_maturity_sensitivity(maturity_years_range)
        self.analyze_sigma_sensitivity(sigma_range)
        self.analyze_ytm_sensitivity(ytm_range)
        
        logger.info("Completed all sensitivity analyses")
        return self.sensitivity_results
    
    def create_sensitivity_plots(self, save_plots: bool = True) -> None:
        """Create comprehensive sensitivity plots."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Callable Bond Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Coupon Rate Sensitivity
        if 'coupon' in self.sensitivity_results:
            ax1 = axes[0, 0]
            df_coupon = self.sensitivity_results['coupon']
            
            ax1.plot(df_coupon['coupon_rate'] * 100, df_coupon['callable_bond_value'], 
                    'b-', linewidth=2, label='Callable Bond')
            ax1.plot(df_coupon['coupon_rate'] * 100, df_coupon['straight_bond_value'], 
                    'r--', linewidth=2, label='Straight Bond')
            
            ax1.set_xlabel('Coupon Rate (%)')
            ax1.set_ylabel('Bond Value')
            ax1.set_title('Sensitivity to Coupon Rate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add option value on secondary y-axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df_coupon['coupon_rate'] * 100, df_coupon['option_value'], 
                         'g:', linewidth=2, label='Option Value')
            ax1_twin.set_ylabel('Option Value', color='g')
            ax1_twin.tick_params(axis='y', labelcolor='g')
        
        # Plot 2: Time to Maturity Sensitivity
        if 'maturity' in self.sensitivity_results:
            ax2 = axes[0, 1]
            df_maturity = self.sensitivity_results['maturity']
            
            ax2.plot(df_maturity['time_to_maturity_years'], df_maturity['callable_bond_value'], 
                    'b-', linewidth=2, label='Callable Bond')
            ax2.plot(df_maturity['time_to_maturity_years'], df_maturity['straight_bond_value'], 
                    'r--', linewidth=2, label='Straight Bond')
            
            ax2.set_xlabel('Time to Maturity (Years)')
            ax2.set_ylabel('Bond Value')
            ax2.set_title('Sensitivity to Time to Maturity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add option value on secondary y-axis
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df_maturity['time_to_maturity_years'], df_maturity['option_value'], 
                         'g:', linewidth=2, label='Option Value')
            ax2_twin.set_ylabel('Option Value', color='g')
            ax2_twin.tick_params(axis='y', labelcolor='g')
        
        # Plot 3: Volatility (Sigma) Sensitivity
        if 'sigma' in self.sensitivity_results:
            ax3 = axes[1, 0]
            df_sigma = self.sensitivity_results['sigma']
            
            ax3.plot(df_sigma['sigma'] * 100, df_sigma['callable_bond_value'], 
                    'b-', linewidth=2, label='Callable Bond')
            ax3.plot(df_sigma['sigma'] * 100, df_sigma['straight_bond_value'], 
                    'r--', linewidth=2, label='Straight Bond')
            
            ax3.set_xlabel('Volatility (%)')
            ax3.set_ylabel('Bond Value')
            ax3.set_title('Sensitivity to Volatility (Sigma)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add option value on secondary y-axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df_sigma['sigma'] * 100, df_sigma['option_value'], 
                         'g:', linewidth=2, label='Option Value')
            ax3_twin.set_ylabel('Option Value', color='g')
            ax3_twin.tick_params(axis='y', labelcolor='g')
        
        # Plot 4: YTM Sensitivity
        if 'ytm' in self.sensitivity_results:
            ax4 = axes[1, 1]
            df_ytm = self.sensitivity_results['ytm']
            
            ax4.plot(df_ytm['straight_bond_ytm'] * 100, df_ytm['callable_bond_value'], 
                    'b-', linewidth=2, label='Callable Bond')
            ax4.plot(df_ytm['straight_bond_ytm'] * 100, df_ytm['straight_bond_value'], 
                    'r--', linewidth=2, label='Straight Bond')
            
            ax4.set_xlabel('Straight Bond YTM (%)')
            ax4.set_ylabel('Bond Value')
            ax4.set_title('Sensitivity to Straight Bond YTM')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add option value on secondary y-axis
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df_ytm['straight_bond_ytm'] * 100, df_ytm['option_value'], 
                         'g:', linewidth=2, label='Option Value')
            ax4_twin.set_ylabel('Option Value', color='g')
            ax4_twin.tick_params(axis='y', labelcolor='g')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('callable_bond_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Sensitivity plots saved as 'callable_bond_sensitivity_analysis.png'")
        
        plt.show()
    
    def create_call_probability_plots(self, save_plots: bool = True) -> None:
        """Create plots showing call probability sensitivity."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Call Probability Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # Plot call probabilities for each sensitivity
        sensitivities = [
            ('coupon', 'coupon_rate', 'Coupon Rate (%)', 100),
            ('maturity', 'time_to_maturity_years', 'Time to Maturity (Years)', 1),
            ('sigma', 'sigma', 'Volatility (%)', 100),
            ('ytm', 'straight_bond_ytm', 'Straight Bond YTM (%)', 100)
        ]
        
        for i, (key, x_col, x_label, scale) in enumerate(sensitivities):
            if key in self.sensitivity_results:
                ax = axes[i // 2, i % 2]
                df = self.sensitivity_results[key]
                
                ax.plot(df[x_col] * scale, df['call_probability'] * 100, 
                       'ro-', linewidth=2, markersize=6)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('Call Probability (%)')
                ax.set_title(f'Call Probability vs {x_label}')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('callable_bond_call_probability_sensitivity.png', dpi=300, bbox_inches='tight')
            logger.info("Call probability plots saved as 'callable_bond_call_probability_sensitivity.png'")
        
        plt.show()
    
    def generate_sensitivity_report(self) -> str:
        """Generate a comprehensive sensitivity analysis report."""
        
        report = []
        report.append("="*80)
        report.append("CALLABLE BOND SENSITIVITY ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Base parameters
        report.append("BASE PARAMETERS:")
        report.append(f"Coupon Rate: {self.base_coupon_rate:.2%}")
        report.append(f"Time to Maturity: {self.base_time_to_maturity:.1f} years")
        report.append(f"Volatility (Sigma): {self.base_sigma:.2%}")
        report.append(f"Straight Bond YTM: {self.base_straight_bond_ytm:.2%}")
        report.append(f"Call Protection: {self.call_protection_years} year(s)")
        report.append(f"Number of Simulations: {self.num_simulations:,}")
        report.append("")
        
        # Individual sensitivity summaries
        for param_name, df_key in [
            ('COUPON RATE', 'coupon'),
            ('TIME TO MATURITY', 'maturity'), 
            ('VOLATILITY (SIGMA)', 'sigma'),
            ('STRAIGHT BOND YTM', 'ytm')
        ]:
            if df_key in self.sensitivity_results:
                df = self.sensitivity_results[df_key]
                report.append(f"{param_name} SENSITIVITY:")
                
                # Find min and max values
                min_callable = df['callable_bond_value'].min()
                max_callable = df['callable_bond_value'].max()
                min_option = df['option_value'].min()
                max_option = df['option_value'].max()
                min_call_prob = df['call_probability'].min()
                max_call_prob = df['call_probability'].max()
                
                report.append(f"  Callable Bond Value Range: {min_callable:.4f} - {max_callable:.4f}")
                report.append(f"  Option Value Range: {min_option:.4f} - {max_option:.4f}")
                report.append(f"  Call Probability Range: {min_call_prob:.1%} - {max_call_prob:.1%}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results_to_csv(self, filename_prefix: str = "callable_bond_sensitivity") -> None:
        """Save all sensitivity results to CSV files."""
        
        try:
            # Save each sensitivity analysis as CSV
            for param_name, df in self.sensitivity_results.items():
                csv_filename = f"{filename_prefix}_{param_name}_results.csv"
                df.to_csv(csv_filename, index=False)
                logger.info(f"Saved {param_name} sensitivity results to {csv_filename}")
            
            # Create and save summary CSV
            summary_data = []
            for param_name, df in self.sensitivity_results.items():
                summary_data.append({
                    'Parameter': param_name.upper(),
                    'Min_Callable_Value': df['callable_bond_value'].min(),
                    'Max_Callable_Value': df['callable_bond_value'].max(),
                    'Min_Option_Value': df['option_value'].min(),
                    'Max_Option_Value': df['option_value'].max(),
                    'Min_Call_Probability': df['call_probability'].min(),
                    'Max_Call_Probability': df['call_probability'].max(),
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv_filename = f"{filename_prefix}_summary.csv"
            summary_df.to_csv(summary_csv_filename, index=False)
            logger.info(f"Saved summary results to {summary_csv_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")


def main():
    """Main function to run the sensitivity analysis."""
    
    # Define base parameters (similar to the main callable bond example)
    issue_date = Date(27, 5, 2025)
    maturity_date = Date(27, 5, 2030)  # 5 years
    base_coupon_rate = 0.046  # 4.6%
    base_straight_bond_ytm = 0.046  # 4.6%
    base_sigma = 0.0095  # 0.95%
    
    # Create sensitivity analyzer
    analyzer = CallableBondSensitivityAnalyzer(
        base_issue_date=issue_date,
        base_maturity_date=maturity_date,
        base_coupon_rate=base_coupon_rate,
        base_straight_bond_ytm=base_straight_bond_ytm,
        base_sigma=base_sigma,
        call_protection_years=1,
        num_simulations=1000,  # Reduced for faster execution
    )
    
    try:
        # Run full sensitivity analysis
        logger.info("Starting comprehensive sensitivity analysis...")
        
        # Define custom ranges for more targeted analysis
        coupon_range = np.arange(0.03, 0.07, 0.005).tolist()  # 3% to 7%
        maturity_range = [3, 4, 5, 6, 7, 8, 10]  # Years
        sigma_range = np.arange(0.005, 0.020, 0.0025).tolist()  # 0.5% to 2%
        ytm_range = np.arange(0.025, 0.065, 0.005).tolist()  # 2.5% to 6.5%
        
        results = analyzer.run_full_sensitivity_analysis(
            coupon_range=coupon_range,
            maturity_years_range=maturity_range,
            sigma_range=sigma_range,
            ytm_range=ytm_range,
        )
        
        # Generate and print report
        report = analyzer.generate_sensitivity_report()
        print(report)
        
        # Create plots
        analyzer.create_sensitivity_plots(save_plots=True)
        analyzer.create_call_probability_plots(save_plots=True)
        
        # Save results to CSV
        analyzer.save_results_to_csv()
        
        # Print detailed results for each sensitivity analysis
        print("\n" + "="*80)
        print("DETAILED SENSITIVITY RESULTS")
        print("="*80)
        
        for param_name, df in results.items():
            print(f"\n{param_name.upper()} SENSITIVITY:")
            print("-" * 40)
            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        logger.info("Sensitivity analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 