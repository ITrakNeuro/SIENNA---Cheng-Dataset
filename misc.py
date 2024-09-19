def print_neat_results(all_results):
    class_labels = {0: "Pituitary", 1: "Meningioma", 2: "Glioma"}
    
    for patient_name, model_results in all_results.items():
        print(f"\n{'='*10} Results for {patient_name} {'='*10}")
        for model_name, class_results in model_results.items():
            print(f"\nModel: {model_name}")
            for class_idx, metrics in class_results.items():
                if isinstance(class_idx, str) and class_idx.startswith("Class_"):
                    class_idx = int(class_idx.split("_")[1])
                
                class_name = class_labels.get(class_idx, f"{class_idx}")
                print(f"\nClass: {class_name}")
                print(f"  Accuracy: {metrics['Accuracy']:.2f}")
                print(f"  Precision: {metrics['Precision']:.2f}")
                print(f"  Recall: {metrics['Recall']:.2f}")
                print(f"  F1 Score: {metrics['F1 Score']:.2f}")
        print(f"\n{'='*30}")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot(all_results):
  
    metrics_data = []
    count_data = []
    for run_name, models in all_results.items():
        for model_name, classes in models.items():
            for class_name, metrics in classes.items():
                for metric, value in metrics.items():
                    if metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']:
                        metrics_data.append({
                            'Run': run_name,
                            'Class': class_name,
                            'Metric': metric,
                            'Value': value
                        })
                    else:
                        count_data.append({
                            'Run': run_name,
                            'Class': class_name,
                            'Metric': metric,
                            'Value': value
                        })

    metrics_df = pd.DataFrame(metrics_data)
    counts_df = pd.DataFrame(count_data)
    custom_palette = sns.color_palette(["#FFCC80", "#FFAB91", "#CE93D8"])

    runs = all_results.keys()
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 4 * len(runs))) 


    for i, run_name in enumerate(runs):
        run_metrics_df = metrics_df[metrics_df['Run'] == run_name]
        run_counts_df = counts_df[counts_df['Run'] == run_name].pivot_table(index=['Class'], columns='Metric', values='Value')
        
        run_counts_df = run_counts_df[['TP', 'TN', 'FP', 'FN']]

        sns.barplot(x='Metric', y='Value', hue='Class', data=run_metrics_df, ax=axes[i, 0], palette=custom_palette)
        axes[i, 0].set_title(f'Performance Metrics - {run_name}')
        axes[i, 0].set_xlabel('Metric')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].tick_params(axis='x', rotation=45)

        run_counts_df.plot(
            kind='bar',
            stacked=True,
            ax=axes[i, 1],
            color=["#236ba6", "#3e99c6", "#FDDE62", "#FF914D"],  
            edgecolor='none',
            width=0.75
        )
        
        axes[i, 1].set_ylabel('Count')
        axes[i, 1].set_xlabel('Class')
        axes[i, 1].set_title(f'Confusion Matrix Components - {run_name}')
        axes[i, 1].legend(title="Metrics", loc='upper right', bbox_to_anchor=(1.15, 1))
        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
