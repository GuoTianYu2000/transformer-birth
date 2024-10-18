hook_no_mlp = check_embed(target_layers=[0, 1, 2], target_heads=[(0, 0), (1, 0), (2, 0)], target_mlp_layers=[])
pred_no_mlp, outputs_list_no_mlp = model.modified_forward_with_hook(x, hook)
probs_no_mlp = get_oracle_predicts(x, ds)
risk_no_mlp = get_risk(probs, pred_no_mlp, predict_in_logits=True, triggers_pos=triggers_pos)
pred_in_probs_no_mlp = F.softmax(pred_no_mlp, dim=-1)
risk_no_mlp[1:3]
