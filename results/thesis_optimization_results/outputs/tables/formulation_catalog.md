| formulation_id | formulation_name | description | uses_ed | uses_line | uses_n1 | uses_surrogate | uses_redispatch | embedding_mode | model_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ed | ED | Economic dispatch only. | 1 | 0 | 0 | 0 | 0 | disabled | MTLSharedHeads |
| ed_line | ED + Line | Economic dispatch with base-case PTDF line limits. | 1 | 1 | 0 | 0 | 0 | disabled | MTLSharedHeads |
| ed_line_n1 | ED + Line + N-1 | Preventive N-1 line-security constraints (LODF), no surrogate. | 1 | 1 | 1 | 0 | 0 | disabled | MTLSharedHeads |
| ed_surrogate | ED + Surrogate | Dynamic-security surrogate embedded, without PTDF/N-1 line constraints. | 1 | 0 | 0 | 1 | 0 | milp | MTLSharedHeads |
| ed_line_n1_surrogate | ED + Line + N-1 + Surrogate | Full preventive formulation used as main thesis baseline. | 1 | 1 | 1 | 1 | 0 | milp | MTLSharedHeads |
| ed_line_n1_surrogate_redispatch | ED + Line + N-1 + Surrogate + Redispatch | Optional sensitivity with N-1 redispatch recourse approximation. | 1 | 1 | 1 | 1 | 1 | milp | MTLSharedHeads |
