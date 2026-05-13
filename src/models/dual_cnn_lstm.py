# -*- coding: utf-8 -*-
"""
Dual CNN-LSTM model for joint SOH/SOC estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42


class SOH_Estimator(nn.Module):
    def __init__(
        self,
        input_channels=3,
        seq_length=46,
        hidden_dim=32,
        aux_dim=2,
        conv_dropout=0.0,
    ):
        super(SOH_Estimator, self).__init__()

        self.aux_dim = aux_dim

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=30,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(30)

        if conv_dropout > 0:
            self.dropout = nn.Dropout(p=conv_dropout)
        else:
            self.dropout = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=30,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
            dropout=0.1,
        )

        self.fusion_fc = nn.Linear(hidden_dim + aux_dim, 1)

    def forward(self, x, aux_info=None):
        batch_size, num_samples, seq_length, input_channels = x.shape

        x = x.view(batch_size * num_samples, seq_length, input_channels)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x.mean(dim=1)

        if aux_info is not None:
            aux_info = aux_info.view(batch_size * num_samples, -1)
        else:
            aux_info = torch.zeros(
                batch_size * num_samples,
                self.aux_dim,
                device=x.device,
                dtype=x.dtype,
            )

        combined = torch.cat([x, aux_info], dim=1)

        soh_output = self.fusion_fc(combined)
        soh_output = F.relu(soh_output)
        soh_output = soh_output.view(batch_size, num_samples, 1)

        return soh_output


class SOC_Estimator(nn.Module):
    def __init__(
        self,
        input_channels=2,
        seq_length=30,
        hidden_dim=32,
        aux_dim=2,
        conv_dropout=0.0,
    ):
        super(SOC_Estimator, self).__init__()

        self.aux_dim = aux_dim

        self.conv1 = nn.Conv1d(
            in_channels=input_channels + 1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=30,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(30)

        if conv_dropout > 0:
            self.dropout = nn.Dropout(p=conv_dropout)
        else:
            self.dropout = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=30,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
            dropout=0.1,
        )

        self.fusion_fc = nn.Linear(hidden_dim + self.aux_dim, 1)

    def forward(self, x, soh_input, aux_info=None):
        batch_size, num_samples, seq_length, input_channels = x.size()

        soh_input = torch.nanmean(soh_input, dim=1)

        soh_feature = soh_input.unsqueeze(1).unsqueeze(2)
        soh_feature = soh_feature.expand(batch_size, num_samples, seq_length, 1)

        x = torch.cat([x, soh_feature], dim=-1)

        x = x.view(batch_size * num_samples, seq_length, input_channels + 1)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x.mean(dim=1)

        if aux_info is not None:
            aux_info = aux_info.view(batch_size * num_samples, self.aux_dim)
        else:
            aux_info = torch.zeros(
                batch_size * num_samples,
                self.aux_dim,
                device=x.device,
                dtype=x.dtype,
            )

        x = torch.cat([x, aux_info], dim=-1)

        soc_raw = self.fusion_fc(x)
        soc_raw = F.relu(soc_raw)
        soc_pred = torch.clamp(soc_raw, 0.0, 1.0)

        soc_pred = soc_pred.view(batch_size, num_samples, 1)

        return soc_pred


class DualCNNLSTMModel(nn.Module):
    def __init__(
        self,
        soh_input_channels=3,
        soc_input_channels=2,
        soh_seq_length=46,
        soc_seq_length=30,
        hidden_dim=32,
        aux_dim=2,
        conv_dropout=0.0,
        soh_aux_start=6,
        soc_aux_start=2,
    ):
        super(DualCNNLSTMModel, self).__init__()

        self.soh_aux_start = soh_aux_start
        self.soc_aux_start = soc_aux_start

        self.soh_model = SOH_Estimator(
            input_channels=soh_input_channels,
            seq_length=soh_seq_length,
            hidden_dim=hidden_dim,
            aux_dim=aux_dim,
            conv_dropout=conv_dropout,
        )

        self.soc_model = SOC_Estimator(
            input_channels=soc_input_channels,
            seq_length=soc_seq_length,
            hidden_dim=hidden_dim,
            aux_dim=aux_dim,
            conv_dropout=conv_dropout,
        )

    def forward(self, data_soh, data_soc):
        x_soh, y_soh = data_soh
        x_soc, y_soc = data_soc

        if y_soh.size(-1) > self.soh_aux_start:
            aux_info_soh = y_soh[:, :, self.soh_aux_start:]
        else:
            aux_info_soh = None

        soh_pred = self.soh_model(x_soh, aux_info_soh)

        if y_soc.size(-1) > self.soc_aux_start:
            aux_info_soc = y_soc[:, :, self.soc_aux_start:]
        else:
            aux_info_soc = None

        soc_pred = self.soc_model(x_soc, soh_pred, aux_info_soc)

        return soh_pred, soc_pred


def init_weights(m, seed=SEED):
    """
    Original-style weight initialization.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.manual_seed(seed)

        nn.init.xavier_normal_(m.weight)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LSTM):
        torch.manual_seed(seed)

        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)