import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import statcast
from datetime import datetime, timedelta
import os
from scipy import stats
import requests
from typing import Dict, List, Tuple
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BettingOddsCollector:
    def __init__(self):
        self.api_key = os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")
        self.base_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
    
    def get_live_odds(self, game_id: str) -> pd.DataFrame:
        """
        Fetch live odds data for a specific game
        """
        endpoint = f"{self.base_url}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads',
            'oddsFormat': 'decimal',
            'eventIds': game_id
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            odds_data = response.json()
            
            if not odds_data:
                return pd.DataFrame()
            
            # Process odds data into a DataFrame
            odds_list = []
            for game in odds_data:
                for bookmaker in game['bookmakers']:
                    for market in bookmaker['markets']:
                        for outcome in market['outcomes']:
                            odds_list.append({
                                'game_id': game['id'],
                                'timestamp': game['commence_time'],
                                'bookmaker': bookmaker['key'],
                                'market': market['key'],
                                'team': outcome['name'],
                                'price': outcome['price'],
                                'point': outcome.get('point', None)
                            })
            
            return pd.DataFrame(odds_list)
        except Exception as e:
            print(f"Error fetching odds data: {e}")
            return pd.DataFrame()
    
    def get_odds_history(self, game_id: str, interval_seconds: int = 30) -> pd.DataFrame:
        """
        Collect odds data at regular intervals for a game
        """
        odds_history = []
        while True:
            odds = self.get_live_odds(game_id)
            if not odds.empty:
                odds['collection_time'] = datetime.now()
                odds_history.append(odds)
            time.sleep(interval_seconds)
        
        return pd.concat(odds_history, ignore_index=True)

class BaseballOddsAnalyzer:
    def __init__(self):
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.odds_collector = BettingOddsCollector()
        
    def get_pitch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch pitch-by-pitch data for the specified date range
        """
        print(f"Fetching Statcast data from {start_date} to {end_date}...")
        pitch_data = statcast(start_date, end_date)
        if pitch_data is not None and not pitch_data.empty:
            print(f"Retrieved {len(pitch_data)} pitch records")
            pitch_data.to_csv(f'{self.data_dir}/pitch_data_{start_date}_{end_date}.csv', index=False)
            return pitch_data
        print("No data available for the specified date range")
        return pd.DataFrame()
    
    def analyze_pitch_impact(self, pitch_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how different pitch outcomes affect the game state
        """
        if pitch_data.empty:
            return pd.DataFrame()
            
        # Convert data types for analysis
        pitch_data['game_pk'] = pitch_data['game_pk'].astype(float)
        pitch_data['balls'] = pitch_data['balls'].astype(float)
        pitch_data['strikes'] = pitch_data['strikes'].astype(float)
        pitch_data['outs_when_up'] = pitch_data['outs_when_up'].astype(float)
            
        # Basic pitch analysis
        pitch_analysis = pitch_data.groupby(['pitch_type', 'description']).agg({
            'game_pk': 'count',
            'balls': 'mean',
            'strikes': 'mean',
            'on_3b': 'mean',
            'on_2b': 'mean',
            'on_1b': 'mean',
            'outs_when_up': 'mean'
        }).reset_index()
        
        # Add advanced statistics
        total_pitches = pitch_analysis['game_pk'].sum()
        pitch_analysis['pitch_frequency'] = pitch_analysis['game_pk'] / total_pitches
        
        # Ensure numeric types
        numeric_columns = ['game_pk', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'pitch_frequency']
        for col in numeric_columns:
            pitch_analysis[col] = pd.to_numeric(pitch_analysis[col], errors='coerce')
        
        return pitch_analysis
    
    def analyze_sequence_impact(self, pitch_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the impact of pitch sequences on game outcomes
        """
        if pitch_data.empty:
            return {}
            
        sequences = []
        for game_id in pitch_data['game_pk'].unique():
            game_pitches = pitch_data[pitch_data['game_pk'] == game_id]
            for i in range(len(game_pitches) - 2):
                sequence = tuple(game_pitches.iloc[i:i+3]['description'].values)
                sequences.append(sequence)
        
        sequence_counts = pd.Series(sequences).value_counts()
        return sequence_counts.head(10).to_dict()
    
    def visualize_pitch_impact(self, analysis_data: pd.DataFrame, sequence_impact: Dict[str, float]):
        """
        Create comprehensive visualizations of pitch impact
        """
        if analysis_data.empty:
            print("No data available for visualization")
            return
            
        # Set the style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Pitch Type Distribution
        ax1 = plt.subplot(2, 2, 1)
        if not analysis_data.empty and 'pitch_type' in analysis_data.columns:
            sns.barplot(data=analysis_data, x='pitch_type', y='pitch_frequency', ax=ax1)
            ax1.set_title('Pitch Type Distribution')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Pitch Count Impact
        ax2 = plt.subplot(2, 2, 2)
        if not analysis_data.empty and all(col in analysis_data.columns for col in ['balls', 'strikes', 'pitch_type']):
            sns.scatterplot(data=analysis_data, x='balls', y='strikes', 
                          hue='pitch_type', size='game_pk', ax=ax2,
                          sizes=(100, 1000), alpha=0.6)
            ax2.set_title('Pitch Count Distribution')
        
        # 3. Pitch Sequence Impact
        ax3 = plt.subplot(2, 2, 3)
        if sequence_impact:
            sequences = list(sequence_impact.keys())
            values = list(sequence_impact.values())
            # Convert sequences to strings for display
            seq_labels = [' -> '.join(str(x) for x in seq) for seq in sequences]
            # Only show top 5 sequences for clarity
            sns.barplot(x=values[:5], y=seq_labels[:5], ax=ax3)
            ax3.set_title('Top 5 Most Common Pitch Sequences')
        else:
            ax3.text(0.5, 0.5, 'No sequence data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        # 4. Pitch Type vs Outcome Heatmap
        ax4 = plt.subplot(2, 2, 4)
        if not analysis_data.empty and all(col in analysis_data.columns for col in ['pitch_type', 'description', 'game_pk']):
            count_data = analysis_data.pivot_table(
                index='pitch_type',
                columns='description',
                values='game_pk',
                fill_value=0
            )
            sns.heatmap(count_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4)
            ax4.set_title('Pitch Type vs Outcome Heatmap')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional visualizations
        self.create_advanced_visualizations(analysis_data)

    def create_advanced_visualizations(self, analysis_data: pd.DataFrame):
        """
        Create additional advanced visualizations
        """
        if analysis_data.empty:
            return
            
        sns.set_theme(style="whitegrid")
        
        # 1. Pitch Type Distribution by Count
        if 'pitch_type' in analysis_data.columns and 'balls' in analysis_data.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=analysis_data, x='pitch_type', y='balls')
            plt.title('Pitch Type Distribution by Ball Count')
            plt.xticks(rotation=45)
            plt.savefig(f'{self.data_dir}/pitch_type_by_count.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Pitch Usage Patterns
        if all(col in analysis_data.columns for col in ['pitch_frequency', 'outs_when_up', 'pitch_type']):
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=analysis_data, x='pitch_frequency', y='outs_when_up', 
                          hue='pitch_type', size='game_pk',
                          sizes=(100, 1000), alpha=0.6)
            plt.title('Pitch Usage Patterns by Game Situation')
            plt.savefig(f'{self.data_dir}/pitch_usage_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_odds_movement(self, pitch_data: pd.DataFrame, odds_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how odds move after different pitch events
        """
        if pitch_data.empty or odds_data.empty:
            return pd.DataFrame()
        
        # Merge pitch and odds data based on timestamp
        pitch_data['timestamp'] = pd.to_datetime(pitch_data['game_date'])
        odds_data['timestamp'] = pd.to_datetime(odds_data['timestamp'])
        
        # Calculate odds changes
        odds_data['odds_change'] = odds_data.groupby(['game_id', 'team', 'bookmaker'])['price'].diff()
        
        # Create windows around pitch events
        window_size = '1min'
        merged_data = pd.merge_asof(
            pitch_data.sort_values('timestamp'),
            odds_data.sort_values('timestamp'),
            on='timestamp',
            by='game_id',
            direction='forward',
            tolerance=pd.Timedelta(window_size)
        )
        
        # Analyze odds movement by pitch type and outcome
        odds_impact = merged_data.groupby(['pitch_type', 'description']).agg({
            'odds_change': ['mean', 'std', 'count'],
            'price': 'mean'
        }).reset_index()
        
        return odds_impact
    
    def visualize_odds_impact(self, odds_impact: pd.DataFrame):
        """
        Create visualizations showing the relationship between pitch events and odds movements
        """
        if odds_impact.empty:
            print("No odds impact data available for visualization")
            return
        
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Average Odds Movement by Pitch Type
        ax1 = plt.subplot(2, 2, 1)
        sns.barplot(data=odds_impact, x='pitch_type', 
                   y=('odds_change', 'mean'), ax=ax1)
        ax1.set_title('Average Odds Movement by Pitch Type')
        ax1.set_xlabel('Pitch Type')
        ax1.set_ylabel('Average Odds Change')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Odds Movement Distribution by Outcome
        ax2 = plt.subplot(2, 2, 2)
        sns.boxplot(data=odds_impact, x='description', 
                   y=('odds_change', 'mean'), ax=ax2)
        ax2.set_title('Odds Movement Distribution by Pitch Outcome')
        ax2.set_xlabel('Pitch Outcome')
        ax2.set_ylabel('Odds Change')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Odds Volatility by Pitch Type
        ax3 = plt.subplot(2, 2, 3)
        sns.scatterplot(data=odds_impact, 
                       x=('odds_change', 'mean'), 
                       y=('odds_change', 'std'),
                       hue='pitch_type',
                       size=('odds_change', 'count'),
                       sizes=(100, 1000),
                       alpha=0.6,
                       ax=ax3)
        ax3.set_title('Odds Volatility vs Average Movement')
        ax3.set_xlabel('Average Odds Change')
        ax3.set_ylabel('Standard Deviation of Odds Change')
        
        # 4. Event Impact Heatmap
        ax4 = plt.subplot(2, 2, 4)
        pivot_data = odds_impact.pivot_table(
            index='pitch_type',
            columns='description',
            values=('odds_change', 'mean'),
            fill_value=0
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', 
                   cmap='RdYlBu', center=0, ax=ax4)
        ax4.set_title('Pitch Event Impact on Odds')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/odds_impact_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def run_live_analysis(self, game_id: str, duration_minutes: int = 60):
        """
        Run live analysis of pitch events and odds movements
        """
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        odds_data = []
        pitch_data = []
        
        while datetime.now() < end_time:
            # Collect current odds
            current_odds = self.odds_collector.get_live_odds(game_id)
            if not current_odds.empty:
                odds_data.append(current_odds)
            
            # Get recent pitches
            current_time = datetime.now()
            start_time = current_time - timedelta(minutes=5)
            recent_pitches = self.get_pitch_data(
                start_time.strftime('%Y-%m-%d'),
                current_time.strftime('%Y-%m-%d')
            )
            if not recent_pitches.empty:
                pitch_data.append(recent_pitches)
            
            # Analyze and visualize
            if odds_data and pitch_data:
                combined_odds = pd.concat(odds_data, ignore_index=True)
                combined_pitches = pd.concat(pitch_data, ignore_index=True)
                odds_impact = self.analyze_odds_movement(combined_pitches, combined_odds)
                self.visualize_odds_impact(odds_impact)
            
            time.sleep(30)  # Wait 30 seconds before next update

def main():
    analyzer = BaseballOddsAnalyzer()
    
    # Set date range for analysis (last 7 days to ensure we get some data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching pitch data from {start_str} to {end_str}...")
    pitch_data = analyzer.get_pitch_data(start_str, end_str)
    
    if pitch_data.empty:
        print("No pitch data available for the specified date range.")
        return
    
    print("Analyzing pitch impact...")
    analysis = analyzer.analyze_pitch_impact(pitch_data)
    
    print("Analyzing pitch sequences...")
    sequence_impact = analyzer.analyze_sequence_impact(pitch_data)
    
    print("Creating visualizations...")
    analyzer.visualize_pitch_impact(analysis, sequence_impact)
    
    # Add odds analysis if API key is available
    if os.getenv('ODDS_API_KEY') and os.getenv('ODDS_API_KEY') != 'your_api_key_here':
        print("Analyzing odds movements...")
        # Get a sample game ID from the pitch data
        if not pitch_data.empty:
            sample_game = pitch_data['game_pk'].iloc[0]
            print(f"Running live odds analysis for game {sample_game}")
            analyzer.run_live_analysis(str(sample_game), duration_minutes=60)
        else:
            print("No game data available for odds analysis")
    else:
        print("Skipping odds analysis - ODDS_API_KEY not set or using default value")
    
    print("Analysis complete! Check the data directory for results.")

if __name__ == "__main__":
    main() 