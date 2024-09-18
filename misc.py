
def print_neat_results(all_results):
    class_labels = {0: "Pituitary", 1: "Meningioma", 2: "Glioma"}
    
    for patient_name, model_results in all_results.items():
        print(f"\n{'='*10} Results for {patient_name} {'='*10}")
        for model_name, class_results in model_results.items():
            print(f"\nModel: {model_name}")
            for class_idx, metrics in class_results.items():
                if isinstance(class_idx, str) and class_idx.startswith("Class_"):
                    class_idx = int(class_idx.split("_")[1])
                
                class_name = class_labels.get(class_idx, f"Unknown class {class_idx}")
                print(f"\nClass: {class_name}")
                print(f"  Accuracy: {metrics['Accuracy']:.2f}")
                print(f"  Precision: {metrics['Precision']:.2f}")
                print(f"  Recall: {metrics['Recall']:.2f}")
                print(f"  F1 Score: {metrics['F1 Score']:.2f}")
        print(f"\n{'='*30}")
