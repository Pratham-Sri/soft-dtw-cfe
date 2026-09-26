import os
import sys
import torch
import numpy as np

from soft_dtw_cfe.config import ALL_DATASETS, DEVICE
from soft_dtw_cfe.data.dataset_loader import load_dataset, get_dataloaders
from soft_dtw_cfe.models.classifier import TSClassifier, train_classifier, save_classifier, load_classifier, evaluate_classifier
from soft_dtw_cfe.models.optuna_tuner import optimise_classifier
from soft_dtw_cfe.methods.proposed_method import SoftDTWCounterfactualGenerator
from soft_dtw_cfe.methods.glacier import GlacierCFE
from soft_dtw_cfe.methods.m_cels import MCELS
from soft_dtw_cfe.evaluation.metrics import evaluate_all_metrics
from soft_dtw_cfe.visualization.plot import plot_counterfactual


def run_experiment_for_dataset(dataset_name: str, skip_optuna: bool = False):
    print(f"\n{'='*60}")
    print(f"Running experiments for {dataset_name}")
    print(f"{'='*60}")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test, metadata = load_dataset(dataset_name)
    train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)
    
    d = metadata["n_channels"]
    c = metadata["n_classes"]
    
    # 2. Train or Load Classifier
    model_path = os.path.join("soft_dtw_cfe", "models", "saved", f"classifier_{dataset_name}.pt")
    
    if os.path.exists(model_path):
        print("Loading existing classifier...")
        model = TSClassifier(n_channels=d, n_classes=c, dropout=0.3)
        model = load_classifier(model, dataset_name)
    else:
        print("Training new classifier...")
        if not skip_optuna:
            print("Optimising hyperparameters...")
            best_params = optimise_classifier(d, c, train_loader, test_loader, n_trials=10) # Reduced for speed
            dropout = best_params["dropout"]
            lr = best_params["lr"]
            wd = best_params["weight_decay"]
        else:
            dropout = 0.3
            lr = 1e-3
            wd = 1e-4
            
        model = TSClassifier(n_channels=d, n_classes=c, dropout=dropout)
        model, best_acc, _ = train_classifier(model, train_loader, test_loader, lr=lr, weight_decay=wd)
        save_classifier(model, dataset_name)
        
    acc = evaluate_classifier(model, test_loader)
    print(f"Classifier Accuracy on {dataset_name}: {acc*100:.2f}%")
    
    # 3. Generate Counterfactuals
    # Select a small subset of test samples to generate CFEs for (for demonstration)
    n_test = min(5, len(X_test)) # Use 5 samples
    X_sample = torch.tensor(X_test[:n_test], dtype=torch.float32)
    y_sample = torch.tensor(y_test[:n_test], dtype=torch.long)
    
    methods = {}
    
    print("\n--- Proposed Method (Soft-DTW) ---")
    proposed = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=200)
    res_proposed = proposed.generate_batch(X_sample, y_sample, X_train, y_train)
    methods["Ours"] = res_proposed
    
    # Only run Glacier on univariate datasets
    if d == 1:
        print("\n--- Glacier Method (Uniform) ---")
        glacier = GlacierCFE(classifier=model, seq_len=metadata["seq_len"], ae_epochs=20, cfe_iterations=200)
        res_glacier = glacier.generate_batch(X_sample, y_sample, X_train, y_train)
        methods["Glacier"] = res_glacier
    else:
        print("\n--- Glacier Method skipped (univariate only) ---")
        
    print("\n--- M-CELS Method ---")
    mcels = MCELS(classifier=model, max_iterations=20)
    res_mcels = mcels.generate_batch(X_sample, y_sample, X_train, y_train)
    methods["M-CELS"] = res_mcels
    
    # 4. Evaluate Metrics
    print(f"\nEvaluation Results for {dataset_name}:")
    for method_name, results in methods.items():
        metrics = evaluate_all_metrics(results, X_train, y_train)
        print(f"\n{method_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    # 5. Visualization (plot one example per method)
    # Get target samples for plotting
    unique_classes = np.unique(y_train)
    target_samples_dict = {}
    for cl in unique_classes:
        mask = (y_train == int(cl))
        target_samples_dict[cl] = torch.tensor(X_train[mask], dtype=torch.float32)
        
    for method_name, results in methods.items():
        if results:
            r = results[0] # plot the first sample
            target_samps = target_samples_dict[int(r["target_class"])]
            save_name = f"{dataset_name}_{method_name}_cf"
            plot_counterfactual(
                r["original"], 
                r["counterfactual"], 
                target_samps,
                dataset_name, 
                method_name, 
                r["original_class"], 
                r["target_class"],
                save_name
            )

def main():
    datasets_to_run = ALL_DATASETS
    
    # Open markdown file for reporting
    report_path = os.path.join(os.path.dirname(__file__), "..", "results_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Soft-DTW Counterfactual Explanations Results\n\n")
        
        for ds in datasets_to_run:
            try:
                # Redirect prints temporarily or just let them go to console
                f.write(f"## Dataset: {ds}\n")
                f.write("| Method | Val | L1 | L2 | DTW | IsoForest |\n")
                f.write("|--------|-----|----|----|-----|-----------|\n")
                
                print(f"\n{'='*60}\nRunning experiments for {ds}\n{'='*60}")
                X_train, y_train, X_test, y_test, metadata = load_dataset(ds)
                train_loader, test_loader = get_dataloaders(X_train, y_train, X_test, y_test)
                
                d = metadata["n_channels"]
                c = metadata["n_classes"]
                
                model_path = os.path.join("soft_dtw_cfe", "models", "saved", f"classifier_{ds}.pt")
                if os.path.exists(model_path):
                    model = TSClassifier(n_channels=d, n_classes=c, dropout=0.3)
                    model = load_classifier(model, ds)
                else:
                    model = TSClassifier(n_channels=d, n_classes=c, dropout=0.3)
                    model, best_acc, _ = train_classifier(model, train_loader, test_loader, lr=1e-3, weight_decay=1e-4)
                    save_classifier(model, ds)
                
                acc = evaluate_classifier(model, test_loader)
                f.write(f"**Classifier Accuracy:** {acc*100:.2f}%\n\n")
                
                n_test = min(5, len(X_test)) # Use 5 samples for speed
                X_sample = torch.tensor(X_test[:n_test], dtype=torch.float32)
                y_sample = torch.tensor(y_test[:n_test], dtype=torch.long)
                
                methods = {}
                proposed = SoftDTWCounterfactualGenerator(classifier=model, num_iterations=100)
                methods["Ours"] = proposed.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
                
                if d == 1:
                    glacier = GlacierCFE(classifier=model, seq_len=metadata["seq_len"], ae_epochs=10, cfe_iterations=100)
                    methods["Glacier"] = glacier.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
                
                mcels = MCELS(classifier=model, max_iterations=10)
                methods["M-CELS"] = mcels.generate_batch(X_sample, y_sample, X_train, y_train, verbose=False)
                
                for method_name, results in methods.items():
                    metrics = evaluate_all_metrics(results, X_train, y_train)
                    f.write(f"| {method_name} | {metrics.get('Val',0):.4f} | {metrics.get('L1',0):.4f} | {metrics.get('L2',0):.4f} | {metrics.get('DTW',0):.4f} | {metrics.get('IsoForest',0):.4f} |\n")
                    
                f.write("\n")
                f.flush()
            except Exception as e:
                print(f"Failed on {ds}: {e}")
                f.write(f"**Failed to evaluate:** {e}\n\n")

if __name__ == "__main__":
    main()
