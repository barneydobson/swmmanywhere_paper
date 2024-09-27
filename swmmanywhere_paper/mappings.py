from __future__ import annotations

metric_mapping = {
"outfall_relerror_length": "mRL",
"outfall_relerror_npipes": "mRP",
"outfall_relerror_nmanholes": "mRM",
"nc_deltacon0": "mD0",
"nc_laplacian_dist": "mLD",
"nc_vertex_edge_distance": "mVD",
"kstest_edge_betweenness": "mKE",
"kstest_betweenness": "mKN",
"outfall_relerror_diameter": "mRD",
"outfall_kstest_diameters": "mKD",
"outfall_nse_flow" : "mNQ",
"outfall_kge_flow": "mKQ",
"outfall_relerror_flow": "mRQ",
"outfall_nse_flooding": "mNF",
"outfall_kge_flooding": "mKF",
"outfall_relerror_flooding": "mRF"
}

param_mapping = {
"node_merge_distance" : "pNM",
"outfall_length" : "pOL",
"max_street_length" : "pXS",
"river_buffer_distance" : "pRB",
"chahinian_slope_scaling" : "pSS",
"chahinian_angle_scaling" : "pAS",
"length_scaling" : "pLS",
"contributing_area_scaling" : "pCS",
"chahinian_slope_exponent" : "pSE",
"chahinian_angle_exponent" : "pAE",
"length_exponent" : "pLE",
"contributing_area_exponent" : "pCE",
"max_fr" : "pFR",
"min_v" : "pMV",
"max_v" : "pXV",
"min_depth" : "pMD",
"max_depth" : "pXD",
"precipitation" : "pDP",
}