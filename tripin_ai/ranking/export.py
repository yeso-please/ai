"""LightGBM 순위 모델을 백엔드(Java)가 읽는 단순 JSON으로 내보낸다.

백엔드는 LightGBM 네이티브 라이브러리 없이 이 파일만으로 점수를 계산한다(backend `RankerModel`).
형식: {"version", "features": [...], "categorical": {필드: {값: 코드}}, "pca": {"mean", "components"},
       "trees": [[{"f", "t", "c"(범주 코드 목록), "dl", "mt", "l", "r", "v"}, ...], ...]}
  - 노드 배열의 0번이 뿌리. 잎은 "v"(잎 값)만 가진다. 자식 "l"/"r"은 같은 배열의 인덱스다.
  - 수치 분할: x <= t 이면 왼쪽. 범주 분할: 코드가 "c"에 있으면 왼쪽, 음수(모르는 값)는 오른쪽.
  - 결측: mt = "NaN"이면 NaN일 때, "Zero"면 0일 때 dl(기본 왼쪽 여부)을 따른다.
점수 = 모든 트리 잎 값의 합(LightGBM `predict(raw_score=True)`와 같다).
"""
import json


def _flatten(tree):
    nodes = []

    def walk(node):
        index = len(nodes)
        nodes.append(None)
        if "leaf_value" in node:
            nodes[index] = {"v": float(node["leaf_value"])}
            return index
        entry = {"f": int(node["split_feature"]), "dl": bool(node["default_left"]),
                 "mt": node.get("missing_type", "None")}
        if node["decision_type"] == "==":
            entry["c"] = [int(x) for x in str(node["threshold"]).split("||")]
        elif node["decision_type"] == "<=":
            entry["t"] = float(node["threshold"])
        else:
            raise ValueError(f"Unsupported decision type {node['decision_type']}")
        entry["l"] = walk(node["left_child"])
        entry["r"] = walk(node["right_child"])
        nodes[index] = entry
        return index

    walk(tree["tree_structure"])
    return nodes


def export_model(booster, path, version, features, categorical=None, pca=None, extra=None):
    dump = booster.dump_model(num_iteration=booster.best_iteration or None)
    if dump["feature_names"] != list(features):
        raise ValueError("Feature order differs from the booster")
    payload = {"version": version, "features": list(features), "categorical": categorical or {},
               "pca": pca, "trees": [_flatten(t) for t in dump["tree_info"]], **(extra or {})}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return payload
