import matplotlib.pyplot as plt
from trade_brown_motion import BondPriceSimulator

if __name__ == "__main__":
    disc_simulator = BondPriceSimulator(
        S0=100,
        mu=0.02,
        sigma=0.015,
        days=365000,
        days_of_year=365,
        num_paths=1,
        mode="discrete",
    )
    prices_df = disc_simulator.generate_prices()
    disc_simulator.verify_normal_distribution()
