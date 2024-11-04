import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class LayerMetrics:
    attention_metrics: Dict[str, float]
    mlp_metrics: Dict[str, float]
    combined_metrics: Dict[str, float]

@dataclass
class StabilityMetrics:
    max_le: float
    mean_le: float
    variance_le: float
    num_positive: int
    spectrum: np.ndarray
    layer_wise_metrics: List[LayerMetrics]

def plot_spectrum(spectrum: np.ndarray, title: str = "Lyapunov Spectrum"):
    plt.figure(figsize=(10, 6))
    plt.plot(spectrum, 'b-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Lyapunov Exponent")
    plt.grid(True)
    plt.show()

class Pythia70mAnalyzer:
    def __init__(self):
        self.model_name = "EleutherAI/pythia-410m"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision="step0").to(self.device)
        # self.model = AutoModelForCausalLM.from_pretrained("models-xinyue/pythia-70m-step140000-to-pythia-410m").to(self.device)
        print(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = len(self.model.gpt_neox.layers)
        print(f"Model loaded successfully: {self.num_layers} layers, {self.hidden_size} hidden size")

    def prepare_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Prepare attention mask for GPTNeoX attention."""
        # Create causal mask
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=self.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Extend attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Combine masks
        extended_attention_mask = extended_attention_mask & ~causal_mask
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask

    def compute_layer_jacobian(self, 
                             layer_idx: int, 
                             hidden_states: torch.Tensor,
                             attention_mask: torch.Tensor,
                             position_ids: torch.Tensor,
                             compute_attention: bool = True) -> torch.Tensor:
        """Compute Jacobian matrix for a specific transformer layer."""
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Prepare attention mask
        extended_attention_mask = self.prepare_attention_mask(attention_mask, seq_len)
        
        # Ensure we're tracking gradients
        hidden_states = hidden_states.detach().clone()
        hidden_states.requires_grad_(True)
        
        layer = self.model.gpt_neox.layers[layer_idx]
        
        if compute_attention:
            # Compute attention Jacobian
            residual = hidden_states
            
            # Make sure intermediate tensors retain gradients
            hidden_states_norm = layer.input_layernorm(hidden_states)
            hidden_states_norm.retain_grad()
            
            attention_output = layer.attention(
                hidden_states_norm,
                attention_mask=extended_attention_mask,
                position_ids=position_ids
            )[0]
            attention_output.retain_grad()
            
            attention_output = layer.post_attention_dropout(attention_output)
            output = residual + attention_output
            output.retain_grad()
        else:
            # Compute MLP Jacobian
            residual = hidden_states
            hidden_states_norm = layer.post_attention_layernorm(hidden_states)
            hidden_states_norm.retain_grad()
            
            mlp_output = layer.mlp(hidden_states_norm)
            mlp_output.retain_grad()
            
            mlp_output = layer.post_mlp_dropout(mlp_output)
            output = residual + mlp_output
            output.retain_grad()
        
        jacobian = torch.zeros(batch_size, 
                             seq_len * self.hidden_size, 
                             seq_len * self.hidden_size, 
                             device=self.device)
        
        for i in range(seq_len * self.hidden_size):
            if i > 0:
                # Clear all gradients
                hidden_states.grad = None
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
            
            # Compute gradient of i-th output with respect to input
            loss = output.view(batch_size, -1)[:, i].sum()
            loss.backward(retain_graph=True)
            
            # Store gradient
            if hidden_states.grad is not None:
                jacobian[:, i, :] = hidden_states.grad.view(batch_size, -1)
            else:
                print(f"Warning: No gradient for index {i}")
                jacobian[:, i, :] = 0
        
        return jacobian

    def calculate_layer_metrics(self, 
                            layer_idx: int, 
                            hidden_states: torch.Tensor,
                            attention_mask: torch.Tensor,
                            position_ids: torch.Tensor,
                            n_iterations: int = 100) -> LayerMetrics:
        """Calculate stability metrics for attention and MLP components."""
        # Get base dimension
        base_dim = self.hidden_size
        
        # Compute Jacobians
        J_attention = self.compute_layer_jacobian(
            layer_idx, 
            hidden_states.clone(),
            attention_mask,
            position_ids,
            compute_attention=True
        )
        
        J_mlp = self.compute_layer_jacobian(
            layer_idx, 
            hidden_states.clone(),
            attention_mask,
            position_ids,
            compute_attention=False
        )
        
        # Ensure matrices can be multiplied
        if J_mlp.shape[1] != J_attention.shape[2]:
            # Reshape if needed
            min_dim = min(J_mlp.shape[1], J_attention.shape[2])
            J_mlp = J_mlp[:, :min_dim, :min_dim]
            J_attention = J_attention[:, :min_dim, :min_dim]
        
        J_combined = torch.matmul(J_mlp, J_attention)
        
        # Compute metrics
        att_metrics = self._compute_component_metrics(J_attention, n_iterations)
        mlp_metrics = self._compute_component_metrics(J_mlp, n_iterations)
        combined_metrics = self._compute_component_metrics(J_combined, n_iterations)
        
        # Ensure all spectra have same shape
        target_shape = (base_dim,)
        for metrics in [att_metrics, mlp_metrics, combined_metrics]:
            if metrics['spectrum'].shape != target_shape:
                metrics['spectrum'] = metrics['spectrum'][:base_dim]
        
        return LayerMetrics(
            attention_metrics=att_metrics,
            mlp_metrics=mlp_metrics,
            combined_metrics=combined_metrics
        )

    def _compute_component_metrics(self, 
                                 jacobian: torch.Tensor, 
                                 n_iterations: int) -> Dict[str, float]:
        """Compute stability metrics for a component using its Jacobian."""
        batch_size = jacobian.shape[0]
        dim = jacobian.shape[1]
        
        Q = torch.eye(dim, device=self.device).expand(batch_size, -1, -1)
        expansion_factors = torch.zeros(batch_size, dim, n_iterations, device=self.device)
        
        for t in range(n_iterations):
            Q = torch.bmm(jacobian, Q)
            Q, R = torch.linalg.qr(Q)
            expansion_factors[:, :, t] = torch.diagonal(R, dim1=1, dim2=2)
        
        lyap_exps = torch.mean(torch.log(torch.abs(expansion_factors)), dim=2)
        lyap_exps = lyap_exps.mean(dim=0).cpu().numpy()
        
        return {
            'max_le': float(lyap_exps.max()),
            'mean_le': float(lyap_exps.mean()),
            'variance_le': float(lyap_exps.var()),
            'num_positive': int((lyap_exps > 0).sum()),
            'spectrum': lyap_exps
        }

    def analyze_initialization(self, 
                             input_text: str,
                             n_iterations: int = 100) -> StabilityMetrics:
        """Analyze initialization stability across all layers."""
        print(f"Processing input text: {input_text[:50]}...")
        
        # Tokenize input and prepare masks
        tokens = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        seq_len = tokens.input_ids.shape[1]
        attention_mask = tokens.attention_mask
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            hidden_states = self.model.gpt_neox.embed_in(tokens.input_ids)
        
        layer_metrics = []
        
        # Analyze each layer
        for layer_idx in range(self.num_layers):
            print(f"Analyzing layer {layer_idx}...")
            metrics = self.calculate_layer_metrics(
                layer_idx, 
                hidden_states, 
                attention_mask,
                position_ids,
                n_iterations
            )
            layer_metrics.append(metrics)
            
            # Update hidden states for next layer
            with torch.no_grad():
                hidden_states = self.model.gpt_neox.layers[layer_idx](
                    hidden_states,
                    attention_mask=self.prepare_attention_mask(attention_mask, seq_len),
                    position_ids=position_ids
                )[0]
        
        # Compute overall model metrics
        final_spectrum = self._compute_final_spectrum(layer_metrics)
        
        return StabilityMetrics(
            max_le=float(final_spectrum.max()),
            mean_le=float(final_spectrum.mean()),
            variance_le=float(final_spectrum.var()),
            num_positive=int((final_spectrum > 0).sum()),
            spectrum=final_spectrum,
            layer_wise_metrics=layer_metrics
        )

    def _compute_final_spectrum(self, layer_metrics: List[LayerMetrics]) -> np.ndarray:
        """Compute overall model spectrum from layer-wise metrics."""
        # Initialize with the first spectrum to get correct shape
        first_spectrum = layer_metrics[0].combined_metrics['spectrum']
        combined_spectrum = np.zeros_like(first_spectrum)
        
        # Add all spectra
        for metrics in layer_metrics:
            # Ensure spectrum is the right shape
            current_spectrum = metrics.combined_metrics['spectrum']
            if current_spectrum.shape != combined_spectrum.shape:
                # Resize spectrum if needed
                current_spectrum = current_spectrum[:combined_spectrum.shape[0]]
                print(f"Warning: Resizing spectrum from {len(current_spectrum)} to {len(combined_spectrum)}")
                
            combined_spectrum += current_spectrum
            
        return combined_spectrum / len(layer_metrics)

def main():
    # Initialize analyzer
    analyzer = Pythia70mAnalyzer()
    
    # Test input
    text = "The quick brown fox jumps over the lazy dog"
    
    # Get metrics
    metrics = analyzer.analyze_initialization(text)
    
    # Print overall metrics
    print("\nOverall Model Metrics:")
    print(f"Maximum LE: {metrics.max_le:.4f}")
    print(f"Mean LE: {metrics.mean_le:.4f}")
    print(f"Variance: {metrics.variance_le:.4f}")
    print(f"Positive LEs: {metrics.num_positive}")
    
    # Print layer-wise metrics
    print("\nLayer-wise Analysis:")
    for i, layer_metrics in enumerate(metrics.layer_wise_metrics):
        print(f"\nLayer {i}:")
        print("Attention Component:")
        print(f"  Max LE: {layer_metrics.attention_metrics['max_le']:.4f}")
        print(f"  Mean LE: {layer_metrics.attention_metrics['mean_le']:.4f}")
        print("MLP Component:")
        print(f"  Max LE: {layer_metrics.mlp_metrics['max_le']:.4f}")
        print(f"  Mean LE: {layer_metrics.mlp_metrics['mean_le']:.4f}")
    
    # Plot the spectrum
    plot_spectrum(metrics.spectrum, "Overall Lyapunov Spectrum")

if __name__ == "__main__":
    main()