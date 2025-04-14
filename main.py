import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import statcast
from datetime import datetime, timedelta
import os
from scipy import stats
from typing import Dict, List, Tuple
import time

class OddsSimulator:
    def __init__(self):
        self.base_odds = 2.0  # Starting odds (even money)
        self.impact_weights = {
            'strikeout': 0.05,
            'walk': -0.03,
            'single': -0.08,
            'double': -0.15,
            'triple': -0.25,
            'home_run': -0.35,
            'hit_into_play': -0.02,
            'field_out': 0.02
        }
        
    def calculate_odds_movement(self, events: List[str], base_odds: float = None) -> List[float]:
        """
        Calculate odds movements based on sequence of events
        """
        if base_odds is None:
            base_odds = self.base_odds
            
        odds_sequence = [base_odds]
        current_odds = base_odds
        
        for event in events:
            # Get the impact weight for this event
            impact = self.impact_weights.get(event, 0)
            
            # Calculate new odds using a logistic function to keep odds reasonable
            odds_change = impact * current_odds
            current_odds = max(1.01, current_odds + odds_change)  # Ensure odds don't go below 1.01
            odds_sequence.append(current_odds)
            
        return odds_sequence
    
    def simulate_game_odds(self, pitch_data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate odds movements for an entire game based on pitch data
        """
        if pitch_data.empty:
            return pd.DataFrame()
            
        # Sort pitch data by game and timestamp
        pitch_data = pitch_data.sort_values(['game_pk', 'game_date'])
        
        odds_data = []
        for game_id in pitch_data['game_pk'].unique():
            game_pitches = pitch_data[pitch_data['game_pk'] == game_id]
            
            # Map pitch events to our event types
            events = []
            for _, pitch in game_pitches.iterrows():
                if pitch['events'] == 'strikeout':
                    events.append('strikeout')
                elif pitch['events'] == 'walk':
                    events.append('walk')
                elif pitch['events'] == 'single':
                    events.append('single')
                elif pitch['events'] == 'double':
                    events.append('double')
                elif pitch['events'] == 'triple':
                    events.append('triple')
                elif pitch['events'] == 'home_run':
                    events.append('home_run')
                elif pitch['description'] == 'hit_into_play':
                    events.append('hit_into_play')
                elif pitch['events'] == 'field_out':
                    events.append('field_out')
                    
            # Calculate odds movements
            odds_sequence = self.calculate_odds_movement(events)
            
            # Create DataFrame with odds movements
            game_odds = pd.DataFrame({
                'game_id': game_id,
                'pitch_index': range(len(odds_sequence)),
                'simulated_odds': odds_sequence
            })
            
            odds_data.append(game_odds)
            
        return pd.concat(odds_data, ignore_index=True)

class BaseballOddsAnalyzer:
    def __init__(self):
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.odds_simulator = OddsSimulator()
        
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
        
        # Calculate odds impact
        odds_data = self.odds_simulator.simulate_game_odds(pitch_data)
        if not odds_data.empty:
            # Calculate average odds change per pitch type
            pitch_data['pitch_index'] = pitch_data.groupby('game_pk').cumcount()
            pitch_odds = pd.merge(
                pitch_data,
                odds_data,
                left_on=['game_pk', 'pitch_index'],
                right_on=['game_id', 'pitch_index']
            )
            odds_impact = pitch_odds.groupby('pitch_type')['simulated_odds'].agg(['mean', 'std']).reset_index()
            pitch_analysis = pd.merge(pitch_analysis, odds_impact, on='pitch_type', how='left')
        
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
        
        # 4. Simulated Odds Impact
        ax4 = plt.subplot(2, 2, 4)
        if not analysis_data.empty and 'mean' in analysis_data.columns:
            sns.barplot(data=analysis_data, x='pitch_type', y='mean', ax=ax4)
            ax4.set_title('Average Simulated Odds Impact by Pitch Type')
            ax4.tick_params(axis='x', rotation=45)
        
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
            
        # 3. Odds Impact Distribution
        if 'mean' in analysis_data.columns and 'std' in analysis_data.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=analysis_data, x='mean', y='std', 
                          hue='pitch_type', size='pitch_frequency',
                          sizes=(100, 1000), alpha=0.6)
            plt.title('Odds Impact Distribution by Pitch Type')
            plt.xlabel('Average Odds Impact')
            plt.ylabel('Standard Deviation of Impact')
            plt.savefig(f'{self.data_dir}/odds_impact_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

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
    
    print("Analysis complete! Check the data directory for results.")

if __name__ == "__main__":
    main() 