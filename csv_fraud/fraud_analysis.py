import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class FraudAnalysisDashboard:
    """
    Comprehensive fraud detection analysis and visualization
    """
    
    def __init__(self, results_csv='fraud_detection_results.csv'):
        """Load the results from fraud detection"""
        self.df = pd.read_csv(results_csv)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def generate_summary_stats(self):
        """Generate key statistics for dashboard"""
        
        total_txns = len(self.df)
        flagged = (self.df['final_decision'] == 'FLAGGED').sum()
        approved = (self.df['final_decision'] == 'APPROVED').sum()
        
        stats = {
            'total_transactions': total_txns,
            'flagged_count': int(flagged),
            'approved_count': int(approved),
            'fraud_rate': round((flagged / total_txns) * 100, 2),
            'approval_rate': round((approved / total_txns) * 100, 2),
            'high_risk_count': int((self.df['risk_level'] == 'HIGH').sum()),
            'medium_risk_count': int((self.df['risk_level'] == 'MEDIUM').sum()),
            'low_risk_count': int((self.df['risk_level'] == 'LOW').sum()),
            'avg_anomaly_score': round(self.df['anomaly_score'].mean(), 3),
            'total_amount_flagged': round(self.df[self.df['final_decision'] == 'FLAGGED']['amount'].sum(), 2),
            'total_amount_approved': round(self.df[self.df['final_decision'] == 'APPROVED']['amount'].sum(), 2)
        }
        
        return stats
    
    def plot_fraud_detection_overview(self, save_path='charts/fraud_overview.png'):
        """Plot overall fraud detection statistics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fraud Detection System Overview', fontsize=16, fontweight='bold')
        
        # 1. Fraud vs Approved Transactions
        decision_counts = self.df['final_decision'].value_counts()
        colors = ['#ff6b6b', '#51cf66']
        axes[0, 0].pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[0, 0].set_title('Final Decision Distribution', fontsize=13, fontweight='bold')
        
        # 2. Risk Level Distribution
        risk_counts = self.df['risk_level'].value_counts()
        risk_colors = {'HIGH': '#e03131', 'MEDIUM': '#fd7e14', 'LOW': '#37b24d'}
        colors_ordered = [risk_colors[level] for level in risk_counts.index]
        axes[0, 1].bar(risk_counts.index, risk_counts.values, color=colors_ordered, edgecolor='black')
        axes[0, 1].set_title('Risk Level Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].set_ylabel('Number of Transactions', fontsize=11)
        axes[0, 1].set_xlabel('Risk Level', fontsize=11)
        for i, v in enumerate(risk_counts.values):
            axes[0, 1].text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 3. Transactions by Decision Over Time
        self.df['date'] = self.df['timestamp'].dt.date
        daily_decisions = self.df.groupby(['date', 'final_decision']).size().unstack(fill_value=0)
        daily_decisions.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                            color=['#51cf66', '#ff6b6b'], edgecolor='black')
        axes[1, 0].set_title('Daily Transaction Decisions', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_xlabel('Date', fontsize=11)
        axes[1, 0].legend(title='Decision', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Anomaly Score Distribution
        flagged_scores = self.df[self.df['final_decision'] == 'FLAGGED']['anomaly_score']
        approved_scores = self.df[self.df['final_decision'] == 'APPROVED']['anomaly_score']
        
        axes[1, 1].hist([approved_scores, flagged_scores], bins=30, label=['Approved', 'Flagged'],
                       color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(-0.6, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
        axes[1, 1].axvline(-0.4, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
        axes[1, 1].set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Anomaly Score', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_rules_analysis(self, save_path='charts/rules_analysis.png'):
        """Analyze which fraud rules are triggered most"""
        
        # Parse triggered rules
        all_rules = []
        for rules in self.df['rule_triggered']:
            if rules != 'NONE':
                all_rules.extend(rules.split(', '))
        
        rule_counts = pd.Series(all_rules).value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Fraud Detection Rules Analysis', fontsize=16, fontweight='bold')
        
        # 1. Rules Triggered Frequency
        axes[0].barh(rule_counts.index, rule_counts.values, color='#4c6ef5', edgecolor='black')
        axes[0].set_title('Frequency of Rules Triggered', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Count', fontsize=11)
        axes[0].set_ylabel('Rule Name', fontsize=11)
        for i, v in enumerate(rule_counts.values):
            axes[0].text(v + 2, i, str(v), va='center', fontweight='bold')
        
        # 2. Multiple Rules Triggered Distribution
        rules_per_txn = self.df['rule_triggered'].apply(
            lambda x: 0 if x == 'NONE' else len(x.split(', '))
        )
        rule_count_dist = rules_per_txn.value_counts().sort_index()
        
        axes[1].bar(rule_count_dist.index, rule_count_dist.values, 
                   color='#f59f00', edgecolor='black', width=0.6)
        axes[1].set_title('Number of Rules Triggered per Transaction', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Number of Rules', fontsize=11)
        axes[1].set_ylabel('Transaction Count', fontsize=11)
        axes[1].set_xticks(range(max(rule_count_dist.index) + 1))
        for i, v in enumerate(rule_count_dist.values):
            axes[1].text(rule_count_dist.index[i], v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_amount_analysis(self, save_path='charts/amount_analysis.png'):
        """Analyze transaction amounts for fraud detection"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Transaction Amount Analysis', fontsize=16, fontweight='bold')
        
        # 1. Amount Distribution by Decision
        flagged_amounts = self.df[self.df['final_decision'] == 'FLAGGED']['amount']
        approved_amounts = self.df[self.df['final_decision'] == 'APPROVED']['amount']
        
        axes[0, 0].boxplot([approved_amounts, flagged_amounts], labels=['Approved', 'Flagged'],
                          patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', edgecolor='black'),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 0].set_title('Transaction Amount by Decision', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Amount (₹)', fontsize=11)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Amount Deviation Ratio Distribution
        axes[0, 1].hist(self.df['amount_deviation_ratio'], bins=50, 
                       color='#845ef7', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(5, color='red', linestyle='--', linewidth=2, 
                          label='HIGH_AMOUNT_SPIKE threshold (5.0)')
        axes[0, 1].axvline(3, color='orange', linestyle='--', linewidth=2, 
                          label='NEW_DEVICE threshold (3.0)')
        axes[0, 1].set_title('Amount Deviation Ratio Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Deviation Ratio', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].set_xlim(0, 15)
        
        # 3. High-Value Flagged Transactions
        high_value_flagged = self.df[(self.df['final_decision'] == 'FLAGGED') & 
                                     (self.df['amount'] > 10000)].sort_values('amount', ascending=False).head(10)
        
        if len(high_value_flagged) > 0:
            axes[1, 0].barh(range(len(high_value_flagged)), high_value_flagged['amount'], 
                           color='#ff6b6b', edgecolor='black')
            axes[1, 0].set_yticks(range(len(high_value_flagged)))
            axes[1, 0].set_yticklabels(high_value_flagged['transaction_id'], fontsize=9)
            axes[1, 0].set_title('Top 10 High-Value Flagged Transactions', fontsize=13, fontweight='bold')
            axes[1, 0].set_xlabel('Amount (₹)', fontsize=11)
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, 'No high-value flagged transactions', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Top 10 High-Value Flagged Transactions', fontsize=13, fontweight='bold')
        
        # 4. Risk Level vs Average Amount
        risk_amounts = self.df.groupby('risk_level')['amount'].mean().sort_values()
        colors_map = {'LOW': '#37b24d', 'MEDIUM': '#fd7e14', 'HIGH': '#e03131'}
        colors = [colors_map[level] for level in risk_amounts.index]
        
        axes[1, 1].bar(risk_amounts.index, risk_amounts.values, color=colors, edgecolor='black')
        axes[1, 1].set_title('Average Transaction Amount by Risk Level', fontsize=13, fontweight='bold')
        axes[1, 1].set_ylabel('Average Amount (₹)', fontsize=11)
        axes[1, 1].set_xlabel('Risk Level', fontsize=11)
        for i, v in enumerate(risk_amounts.values):
            axes[1, 1].text(i, v + 50, f'₹{v:.0f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_behavioral_patterns(self, save_path='charts/behavioral_patterns.png'):
        """Analyze behavioral patterns in fraud detection"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Behavioral Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Transactions in Last 10 Minutes
        axes[0, 0].hist([
            self.df[self.df['final_decision'] == 'APPROVED']['transactions_last_10min'],
            self.df[self.df['final_decision'] == 'FLAGGED']['transactions_last_10min']
        ], bins=range(0, 15), label=['Approved', 'Flagged'], 
           color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(4, color='red', linestyle='--', linewidth=2, label='Threshold (4)')
        axes[0, 0].set_title('Transactions in Last 10 Minutes', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Transaction Count', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].legend(fontsize=10)
        
        # 2. Time Since Last Transaction
        axes[0, 1].hist([
            self.df[self.df['final_decision'] == 'APPROVED']['time_since_last_txn_min'],
            self.df[self.df['final_decision'] == 'FLAGGED']['time_since_last_txn_min']
        ], bins=30, label=['Approved', 'Flagged'], 
           color=['#51cf66', '#ff6b6b'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Time Since Last Transaction', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Minutes', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].set_xlim(0, 300)
        
        # 3. Device Change Impact
        device_change_impact = pd.crosstab(self.df['device_change_flag'], 
                                           self.df['final_decision'], normalize='index') * 100
        device_change_impact.plot(kind='bar', ax=axes[1, 0], color=['#51cf66', '#ff6b6b'], 
                                 edgecolor='black')
        axes[1, 0].set_title('Device Change Impact on Decisions', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Device Changed (0=No, 1=Yes)', fontsize=11)
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 0].legend(title='Decision', fontsize=10)
        axes[1, 0].set_xticklabels(['No Change', 'Changed'], rotation=0)
        
        # 4. Location Change Impact
        location_change_impact = pd.crosstab(self.df['location_change_flag'], 
                                             self.df['final_decision'], normalize='index') * 100
        location_change_impact.plot(kind='bar', ax=axes[1, 1], color=['#51cf66', '#ff6b6b'], 
                                   edgecolor='black')
        axes[1, 1].set_title('Location Change Impact on Decisions', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Location Changed (0=No, 1=Yes)', fontsize=11)
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=11)
        axes[1, 1].legend(title='Decision', fontsize=10)
        axes[1, 1].set_xticklabels(['Same Location', 'Changed'], rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def generate_detailed_report(self, output_path='fraud_detection_report.txt'):
        """Generate a detailed text report"""
        
        stats = self.generate_summary_stats()
        
        report = f"""
{'='*70}
                FRAUD DETECTION SYSTEM - ANALYSIS REPORT
{'='*70}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
1. OVERALL STATISTICS
{'='*70}

Total Transactions Analyzed:     {stats['total_transactions']:,}
Flagged as Fraud:                {stats['flagged_count']:,} ({stats['fraud_rate']}%)
Approved Transactions:           {stats['approved_count']:,} ({stats['approval_rate']}%)

Total Amount Flagged:            ₹{stats['total_amount_flagged']:,.2f}
Total Amount Approved:           ₹{stats['total_amount_approved']:,.2f}

Average Anomaly Score:           {stats['avg_anomaly_score']}

{'='*70}
2. RISK LEVEL BREAKDOWN
{'='*70}

HIGH Risk Transactions:          {stats['high_risk_count']:,}
MEDIUM Risk Transactions:        {stats['medium_risk_count']:,}
LOW Risk Transactions:           {stats['low_risk_count']:,}

{'='*70}
3. FRAUD DETECTION RULES PERFORMANCE
{'='*70}
"""
        
        # Rule analysis
        all_rules = []
        for rules in self.df['rule_triggered']:
            if rules != 'NONE':
                all_rules.extend(rules.split(', '))
        
        rule_counts = pd.Series(all_rules).value_counts()
        
        for rule, count in rule_counts.items():
            percentage = (count / len(self.df)) * 100
            report += f"\n{rule:<30} {count:>6} ({percentage:>5.2f}%)"
        
        report += f"""

{'='*70}
4. KEY INSIGHTS
{'='*70}

- Most Common Fraud Pattern: {rule_counts.index[0] if len(rule_counts) > 0 else 'N/A'}
- Average Amount (Flagged):  ₹{self.df[self.df['final_decision'] == 'FLAGGED']['amount'].mean():.2f}
- Average Amount (Approved): ₹{self.df[self.df['final_decision'] == 'APPROVED']['amount'].mean():.2f}

- Device Change Fraud Rate:  {(self.df[(self.df['device_change_flag'] == 1) & (self.df['final_decision'] == 'FLAGGED')].shape[0] / max(self.df[self.df['device_change_flag'] == 1].shape[0], 1) * 100):.2f}%
- Location Change Fraud Rate: {(self.df[(self.df['location_change_flag'] == 1) & (self.df['final_decision'] == 'FLAGGED')].shape[0] / max(self.df[self.df['location_change_flag'] == 1].shape[0], 1) * 100):.2f}%

{'='*70}
END OF REPORT
{'='*70}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Saved: {output_path}")
        return report
    
    def export_summary_json(self, output_path='fraud_summary.json'):
        """Export summary statistics as JSON for web interface"""
        
        stats = self.generate_summary_stats()
        
        # Add additional metrics
        stats['risk_distribution'] = {
            'HIGH': int((self.df['risk_level'] == 'HIGH').sum()),
            'MEDIUM': int((self.df['risk_level'] == 'MEDIUM').sum()),
            'LOW': int((self.df['risk_level'] == 'LOW').sum())
        }
        
        # Rule performance
        all_rules = []
        for rules in self.df['rule_triggered']:
            if rules != 'NONE':
                all_rules.extend(rules.split(', '))
        
        rule_counts = pd.Series(all_rules).value_counts()
        stats['rule_performance'] = rule_counts.to_dict()
        
        # Daily trends
        self.df['date'] = self.df['timestamp'].dt.date.astype(str)
        daily_stats = self.df.groupby('date').agg({
            'transaction_id': 'count',
            'final_decision': lambda x: (x == 'FLAGGED').sum()
        }).to_dict('index')
        
        stats['daily_trends'] = daily_stats
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved: {output_path}")
        return stats
    
    def generate_all_analysis(self):
        """Generate complete analysis package"""
        
        import os
        os.makedirs('charts', exist_ok=True)
        
        print("\n" + "="*70)
        print("       GENERATING COMPREHENSIVE FRAUD ANALYSIS")
        print("="*70 + "\n")
        
        print("📊 Generating overview charts...")
        self.plot_fraud_detection_overview()
        
        print("📊 Generating rules analysis...")
        self.plot_rules_analysis()
        
        print("📊 Generating amount analysis...")
        self.plot_amount_analysis()
        
        print("📊 Generating behavioral patterns...")
        self.plot_behavioral_patterns()
        
        print("📝 Generating detailed report...")
        report = self.generate_detailed_report()
        
        print("💾 Exporting JSON summary...")
        self.export_summary_json()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated Files:")
        print("  - charts/fraud_overview.png")
        print("  - charts/rules_analysis.png")
        print("  - charts/amount_analysis.png")
        print("  - charts/behavioral_patterns.png")
        print("  - fraud_detection_report.txt")
        print("  - fraud_summary.json")
        print("\n" + "="*70)
        
        # Print summary stats
        stats = self.generate_summary_stats()
        print(f"\n📈 Quick Summary:")
        print(f"   Total Transactions: {stats['total_transactions']:,}")
        print(f"   Fraud Rate: {stats['fraud_rate']}%")
        print(f"   High Risk: {stats['high_risk_count']:,}")
        print(f"   Amount Flagged: ₹{stats['total_amount_flagged']:,.2f}")
        print("="*70 + "\n")


# ============================================
# RUN ANALYSIS
# ============================================

if __name__ == "__main__":
    # Create analysis dashboard
    dashboard = FraudAnalysisDashboard('fraud_detection_results.csv')
    
    # Generate all analysis and visualizations
    dashboard.generate_all_analysis()