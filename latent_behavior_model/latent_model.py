import torch
import torch.nn as nn


# GRU-based latent behavior model
class LatentBehaviorModel(nn.Module):
    def __init__(self, input_dim, event_dim, context_dim, hidden_dim, n_states):
        super().__init__()

        self.n_states = n_states

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.inference_head = nn.Linear(hidden_dim, n_states)

        self.transition_net = nn.Sequential(
            nn.Linear(n_states + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_states)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_states + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, event_dim)
        )

    def forward(self, x, context):
        h, _ = self.encoder(x)

        q_logits = self.inference_head(h)
        q_probs = torch.softmax(q_logits, dim=-1)

        decoder_input = torch.cat([q_probs, context], dim=-1)
        recon_event = self.decoder(decoder_input)

        return {
            "h": h,
            "q_logits": q_logits,
            "q_probs": q_probs,
            "recon_event": recon_event
        }

    def transition_prior(self, prev_q, context_t):
        trans_input = torch.cat([prev_q, context_t], dim=-1)
        prior_logits = self.transition_net(trans_input)
        prior_probs = torch.softmax(prior_logits, dim=-1)

        return prior_logits, prior_probs