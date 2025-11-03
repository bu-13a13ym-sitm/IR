#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math
import time
import numpy as np
import math
from geometry_msgs.msg import Point
import heapq

GRID_SIZE = 20                  # グリッドの分割数 (10x10)
X_MIN = 2.2                    # 障害物エリアの最小X座標 (環境依存)
X_MAX = 2.9                     # 障害物エリアの最大X座標 (環境依存)
Y_MIN = 2.0                     # 障害物エリアの最小Y座標 (standby_2aのYと仮定)
Y_MAX = 3.3                     # 障害物エリアの最大Y座標 (go_throw_2aの手前Yと仮定)
ROBOT_RADIUS = 0.1 / 2         # ロボットの半幅（安全距離のベース）
OBJECT_INFLATION = 0.01


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = []
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 1
    TROFAST_Y_OFFSET = 0.2

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list   = []

        self.start = time.time()
        self.task1_timelim = self.start + 900
        self.task2_timelim = self.start + 1200
        self.timestamp = np.array([self.start], dtype=float)
        self.task1_ave = None

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        itemLabels = self.load_json(self.get_path(["config", "itemLabels.json"]))["itemLabels"]
        self.shape = itemLabels["shape"]
        self.tool = itemLabels["tool"]
        self.food = itemLabels["food"]
        self.kitchen = itemLabels["kitchen"]
        self.task = itemLabels["task"]

        self.grasp_try_cnt = {}
        self.trayA_cnt = 0

        self.__class__.IGNORE_LIST.extend(self.shape)
        self.__class__.IGNORE_LIST.extend(self.tool)

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str  = ""
        for obj in obj_list:
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h )
            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        # objが一つもない場合は、Noneを返す
        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff   = abs(320- gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / xy_diff

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文から対象となる物体名称を抽出する
        """
        #TODO: 関数は未完成です。引数のinstructionを利用すること
        rospy.loginfo("[extract_target_obj_and_person] instruction:"+  instruction)

        # 物体名のキーワードリスト
        OBJECT_KEYWORDS = [
            "cheez-it", "sugar", "chocolate", "gelatin", "meat", "coffee", "tuna", 
            "pringles", "mustard", "soup", "banana", "strawberry", "apple", "lemon",
            "peach", "pear", "plum", "orange", 
        ]
        PERSON_KEYWORDS = ["left", "right"]

        target_obj = None
        target_person = None
        OBJECT_ID_MAPPING = {
            "cheez-it": "cheez_it_box",
            "sugar": "sugar_box",
            "chocolate": "pudding_box",
            "gelatin": "gelatin_box",
            "meat": "potted_meat_can",
            "coffee": "master_chef_can",
            "tuna": "tuna_fish_can",
            "pringles": "chips_can",
            "mustard": "mustard_bottle",
            "soup": "tomato_soup_can",
            "banana": "banana",
            "strawberry": "strawberry",
            "apple": "apple",
            "lemon": "lemon",
            "peach": "peach",
            "pear": "pear",
            "plum": "plum",
            "orange": "orange",
        }
        processed_instruction = instruction.lower()

        for keyword in OBJECT_KEYWORDS:
            if keyword in processed_instruction:
                target_obj = keyword
                break

        # 2. 抽出されたキーワードを実際の物体IDに変換
        if target_obj in OBJECT_ID_MAPPING:
            # 実際のロボットの認識ID（例: "potted_meat_can"）に置き換える
            final_object_id = OBJECT_ID_MAPPING[target_obj]
        else:
            # マッピングにない場合は抽出されたキーワードをそのまま利用するか、エラー処理を行う
            final_object_id = target_obj

        for keyword in PERSON_KEYWORDS:
            if keyword in processed_instruction:
                target_person = keyword
                break

        # target_personの名称を統一
        if target_person == "left":
            target_person = "person_a"
        elif target_person == "right":
            target_person = "person_b"

        if target_obj is None:
            rospy.logwarn("  -> WARNING: Could not extract target object.")
        if target_person is None:
            rospy.logwarn("  -> WARNING: Could not extract target person.")

        return final_object_id, target_person

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        gripper.command(1)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose( grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label=""):
        """
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
            if label == "bowl":
                grasp_pos.x -= 0.05
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        method(grasp_pos)
        return True

    def put_in_place(self, place, into_pose):
        # 指定場所に入れ、all_neutral姿勢を取る。
        self.change_pose("look_at_near_floor")
        a = "go_palce" # TODO 不要な変数
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(
            x, y + self.TROFAST_Y_OFFSET * 1.2 if z > 0.5 else y + self.TROFAST_Y_OFFSET * 1.9, z,
            yaw,  pitch, roll
            )
        gripper.command(1)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose(pos_x, pos_y + self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(detected_objs.bboxes, target_obj)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object to grasp. task2b is aborted.")
            return

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name(target_person)    # TODO: 配達先が固定されているので修正
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        # blockを避ける
        for i in range(3):
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in bboxes]
            waypoint = self.select_next_waypoint(i, pos_bboxes)
            # TODO メッセージを確認するためコメントアウトを外す
            # rospy.loginfo(waypoint)
            self.goto_pos(waypoint)

    def _calculate_yaw(self, start_x, start_y, target_x, target_y):
        """
        開始点と目標点から、目標方向へのYaw角（度数）を計算する。
        """
        delta_x = target_x - start_x
        delta_y = target_y - start_y
        
        # atan2でラジアンを計算
        rad = math.atan2(delta_y, delta_x)
        
        # 度数に変換
        yaw_degrees = math.degrees(rad)
        
        return yaw_degrees

    def _create_cost_map(self, pos_bboxes):
        """
        10x10のコストマップを生成し、各ブロックのコストを返す。
        コストはオブジェクトからの距離の逆数（近いほど高コスト）。
        """
        cost_map = np.zeros((GRID_SIZE, GRID_SIZE))
        grid_res_x = (X_MAX - X_MIN) / GRID_SIZE
        grid_res_y = (Y_MAX - Y_MIN) / GRID_SIZE
        
        # 検出されたオブジェクト座標をPointオブジェクトのリストとして使用
        obj_points = [[p.x, p.y] for p in pos_bboxes]

        for i in range(GRID_SIZE):  # Y軸 (行)
            for j in range(GRID_SIZE):  # X軸 (列)
                center_x = X_MIN + (j + 0.5) * grid_res_x
                center_y = Y_MIN + (i + 0.5) * grid_res_y
                
                min_dist = float('inf')
                
                for obj_x, obj_y in obj_points:
                    dist = math.sqrt((center_x - obj_x)**2 + (center_y - obj_y)**2)
                    min_dist = min(min_dist, dist)
                
                # コストの計算: オブジェクトに近いほど高コスト
                # 安全距離内にオブジェクトがある場合は非常に高い衝突コスト (A*で確実に回避させる)
                if min_dist < ROBOT_RADIUS + OBJECT_INFLATION:
                    cost_map[i, j] = 1000.0  # 衝突リスクあり (高コスト)
                else:
                    # 距離の逆数を使用 (遠いほど低コスト)
                    # 0除算やオーバーフローを避けるために分母をチェック
                    denominator = min_dist - (ROBOT_RADIUS + OBJECT_INFLATION)
                    if denominator <= 0.05: # 安全距離からごく近い場合
                         cost_map[i, j] = 50.0 
                    else:
                         cost_map[i, j] = 1.0 / denominator 
                         
        return cost_map, grid_res_x, grid_res_y

    # Helper function to convert grid index (i, j) to map coordinates (x, y)
    def _grid_to_map(self, i, j, grid_res_x, grid_res_y):
        x = X_MIN + (j + 0.5) * grid_res_x
        y = Y_MIN + (i + 0.5) * grid_res_y
        return x, y

    # Helper function to convert map coordinates (x, y) to grid index (i, j)
    def _map_to_grid(self, x, y, grid_res_x, grid_res_y):
        if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
            return None, None
        j = int((x - X_MIN) / grid_res_x)
        i = int((y - Y_MIN) / grid_res_y)
        # Ensure indices are within bounds
        j = np.clip(j, 0, GRID_SIZE - 1)
        i = np.clip(i, 0, GRID_SIZE - 1)
        return i, j
    
    def _find_safest_path(self, cost_map, grid_res_x, grid_res_y, start_x):
        """
        コストマップ上で、各Y行（i）で最低コストのブロックを選び、最終経路を決定する（貪欲法）。
        """
        waypoints = []
        
        # スタートグリッドインデックスを取得
        _, start_j = self._map_to_grid(start_x, Y_MIN, grid_res_x, grid_res_y)
        current_j = start_j if start_j is not None else GRID_SIZE // 2
        
        # 最初のY行はスタート位置のX座標に近い低コストなブロックを選ぶ
        
        for i in range(GRID_SIZE):  # Y軸をY_MINからY_MAXへ進む
            min_cost = float('inf')
            best_j = -1
            
            # 探索ウィンドウ（移動の滑らかさのために、現在のXから大きく離れない範囲）
            search_min_j = max(0, current_j - 2)
            search_max_j = min(GRID_SIZE - 1, current_j + 2)
            
            # 最低コストのブロックを選択
            for j in range(search_min_j, search_max_j + 1): # X軸をスキャン
                
                # 滑らかさ（移動コスト）を加味
                travel_cost = abs(j - current_j) * 0.5 # X軸移動が大きいほどペナルティ
                total_cost = cost_map[i, j] + travel_cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_j = j
                    
            if best_j == -1:
                rospy.logwarn(f"No safe or reachable block found in row Y={i}. Using current X.")
                # 経路が見つからなかった場合、現在のXを維持してY軸を強制的に進める
                best_j = current_j
                
            # ウェイポイントを追加
            map_x, map_y = self._grid_to_map(i, best_j, grid_res_x, grid_res_y)
            waypoints.append([map_x, map_y])
            current_j = best_j

        # 最終目標点を追加（最後のX座標を維持）
        waypoints.append([waypoints[-1][0], Y_MAX])
        
        return waypoints
    
    # A*探索のためのヒューリスティック関数

    def _get_heuristic(self, x1, y1, x2, y2):
        """
        ヒューリスティックコスト (h): ノード(x1, y1)からゴール(x2, y2)までのユークリッド距離
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _find_optimal_path(self, cost_map, grid_res_x, grid_res_y, start_x):
        """
        コストマップ上でA*探索を実行し、スタートからゴールまでの最適経路を返す。
        """
        rospy.loginfo("Starting A* search...")
        
        # ゴール座標（Y_MAXの中央X）
        # Y_MAXは障害物エリアの境界なので、ゴールセルはY_MAXに近いセルの中央とする
        goal_y_index = GRID_SIZE - 1
        
        # スタートノードの特定
        start_i, start_j = self._map_to_grid(start_x, Y_MIN, grid_res_x, grid_res_y)
        if start_i is None or start_j is None:
            rospy.logerr("Start coordinate is outside the map boundary.")
            return []
            
        start_node = (start_i, start_j)

        # ゴールノードの決定 (マップの最奥Y行で、最もコストの低いセルを仮のゴールとする)
        min_goal_cost = float('inf')
        best_goal_j = start_j # ゴールがない場合はスタートと同じX座標を維持
        
        for j in range(GRID_SIZE):
            if cost_map[goal_y_index, j] < min_goal_cost:
                min_goal_cost = cost_map[goal_y_index, j]
                best_goal_j = j
                
        goal_node = (goal_y_index, best_goal_j)
        
        goal_x, goal_y = self._grid_to_map(goal_node[0], goal_node[1], grid_res_x, grid_res_y)

        # A*に必要なデータ構造の初期化
        # priority queue: (f_cost, i, j)
        open_list = [(0.0, start_i, start_j)]
        
        # g_cost[i][j]: スタートからノード(i, j)までの最小累積コスト
        g_cost = np.full((GRID_SIZE, GRID_SIZE), float('inf'))
        g_cost[start_i, start_j] = 0.0
        
        # parent[i][j]: 最適経路を再構築するための親ノード (pi, pj)
        parent = {}
        
        # 8方向移動の定義 (di, dj)
        # 上, 下, 左, 右, 左上, 右上, 左下, 右下
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # A*メインループ
        while open_list:
            f_cost, i, j = heapq.heappop(open_list)
            current_node = (i, j)

            # ゴールに到達したら終了
            if current_node == goal_node:
                rospy.loginfo("A* path found.")
                break

            # 隣接ノードの探索
            for di, dj in movements:
                ni, nj = i + di, j + dj
                neighbor_node = (ni, nj)
                
                # 境界チェック
                if not (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE):
                    continue

                # 移動コスト計算 (斜めはルート2倍)
                move_distance = math.sqrt(di**2 + dj**2)
                
                # 隣接ノードの障害物コストを取得
                obstacle_cost = cost_map[ni, nj]
                
                # 障害物コストが高い（衝突リスク）なら、経路として使用しない
                if obstacle_cost >= 1000: # 1000は_create_cost_mapで設定した衝突リスクのコスト
                    continue 

                # g(n') = g(n) + move_cost
                new_g_cost = g_cost[i, j] + move_distance + obstacle_cost
                
                # より良い経路が見つかった場合
                if new_g_cost < g_cost[ni, nj]:
                    # gコストを更新
                    g_cost[ni, nj] = new_g_cost
                    
                    # 親ノードを記録
                    parent[neighbor_node] = current_node
                    
                    # hコストを計算
                    nx, ny = self._grid_to_map(ni, nj, grid_res_x, grid_res_y)
                    h_cost = self._get_heuristic(nx, ny, goal_x, goal_y)
                    
                    # fコストを計算
                    new_f_cost = new_g_cost + h_cost
                    
                    # Open Listに追加
                    heapq.heappush(open_list, (new_f_cost, ni, nj))

        # 経路の再構築
        if current_node != goal_node:
            rospy.logwarn("A* failed to find a path to the goal.")
            return []

        waypoints = []
        node = goal_node
        while node in parent:
            map_x, map_y = self._grid_to_map(node[0], node[1], grid_res_x, grid_res_y)
            waypoints.append([map_x, map_y])
            node = parent[node]
            
        # スタートノードは既にgoto_name("standby_2a")で処理されているため省略可能
        # map_x, map_y = self._grid_to_map(start_node[0], start_node[1], grid_res_x, grid_res_y)
        # waypoints.append([map_x, map_y])
        
        waypoints.reverse()
        
        # 最終目標点 (go_throw_2a) を追加
        final_x, final_y, final_yaw = self.coordinates["positions"]["go_throw_2a"]
        waypoints.append([final_x, final_y])

        return waypoints

    def select_next_waypoint(self, pos_bboxes):
        """
        壁際を避け、最も広いオブジェクト間の隙間（垂直二等分線X座標）と
        オブジェクトをクリアした中間Y座標を返す。
        """
        GOAL_Y = 3.3
        
        # --- 1. オブジェクト間 X座標の計算（壁際の排除）---
        
        obj_x_list = sorted([bbox.x for bbox in pos_bboxes])
        gaps = []

        # オブジェクトが1つ以下の場合、オブジェクト間は存在しない
        if len(obj_x_list) < 2:
            rospy.logwarn("Only one or zero objects detected. Using default center X.")
            if obj_x_list:
                # 唯一のオブジェクトが存在する場合、その左右に安全な隙間があると仮定
                # ただし、要件を満たさないため、中央を試す
                center_x = obj_x_list[0] 
            else:
                # オブジェクトなしの場合
                center_x = 2.15 
            
            # X, Y座標を決定せず、中間Y座標のみを計算して、最終段階でY軸移動に強制
            return [center_x, 2.5] # X座標は仮、Y座標は中間点

        # 連続するオブジェクトの中心間を結ぶ隙間のみを評価 (iとi+1の間)
        for i in range(len(obj_x_list) - 1):
            x1 = obj_x_list[i]
            x2 = obj_x_list[i+1]
            
            gap_width = x2 - x1
            center_x = (x1 + x2) / 2 # 垂直二等分線 X 座標
            
            gaps.append((gap_width, center_x))

        # 最も広いオブジェクト間の隙間を選択
        best_gap = max(gaps, key=lambda item: item[0])
        next_waypoint_x = best_gap[1] # 垂直二等分線 X 座標
        
        # --- 2. オブジェクトをクリアする Y座標の計算 ---
        SAFETY_MARGIN_Y = 0.30 
        
        # すべてのオブジェクトの中から最も奥のY座標を取得 (最も安全な中間Y座標を決定するため)
        obj_y_list = [bbox.y for bbox in pos_bboxes]
        
        farthest_y = max(obj_y_list)
        intermediate_y = farthest_y + SAFETY_MARGIN_Y
        
        # 中間Y座標はGOAL_Yを超えないようにクリップ
        next_waypoint_y = min(intermediate_y, GOAL_Y)
        
        rospy.loginfo("Selected Bisector X: {:.2f} (Gap width: {:.2f}m)".format(
            next_waypoint_x, best_gap[0]))
        
        # [垂直二等分線X座標, 中間Y座標] を返す (Yawは移動時に計算)
        return [next_waypoint_x, next_waypoint_y]

    def execute_task1(self):
        """
        task1を実行する
        """        
        rospy.loginfo("#### start Task 1 ####")
        """
        self.pull_out_trofast(0.175, -0.31, 0.285, -90, 100, 0)
        self.pull_out_trofast(0.178, -0.31, 0.565, -90, 100, 0)
        self.pull_out_trofast(0.50, -0.35, 0.285, -90, 100, 0)
        """
        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("floor_nearby_long_table_r", "look_at_floor_nearby_long_table_r"),
            ("long_table_r", "look_at_long_table_r")
        ]

        total_cnt = 0
        for plc, pose in hsr_position:
            while True:
                if not self.continue_task1():
                    return
                # 移動と視線指示
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                # 把持対象の有無チェック
                detected_objs = self.get_latest_detection()
                graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)

                if graspable_obj is None:
                    rospy.logwarn("Cannot determine object to grasp. Grasping is aborted.")
                    break

                label = graspable_obj["label"]
                if label == "tuna_fish_can":
                    label = "bowl"
                elif label == "potted_meat_can":
                    label = "lego_duplo"
                elif label == "sugar_box":
                    label = "rubiks_cube"
                grasp_bbox = graspable_obj["bbox"]
                # TODO ラベル名を確認するためにコメントアウトを外す
                rospy.loginfo("grasp the " + label)
                if label in self.grasp_try_cnt.keys():
                    self.grasp_try_cnt[label] += 1
                    if self.grasp_try_cnt[label] > 3:
                        self.__class__.IGNORE_LIST.append(label)
                        continue
                else:
                    self.grasp_try_cnt[label] = 0

                # 把持対象がある場合は把持関数実施
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                self.change_pose("grasp_on_table")
                self.exec_graspable_method(grasp_pos, label)
                self.change_pose("all_neutral")

                if label in self.shape:
                    self.put_in_place("drawer", "put_in_drawer_left")
                elif label in self.tool:
                    self.put_in_place("stair_like_drawer", "put_in_drawer_bottom")
                elif label in self.food:
                    if label in self.grasp_try_cnt.keys():
                        self.trayA_cnt -= 1
                    if self.trayA_cnt < 3:
                        self.put_in_place("long_table_r_trayA", "put_on_tray")
                    else:
                        self.put_in_place("long_table_r_trayB", "put_on_tray")
                    self.trayA_cnt += 1
                elif label in self.kitchen:
                    self.put_in_place("long_table_r_containerA", "put_in_container")
                elif label in self.task:
                    self.put_in_place("bin_a_place", "put_in_bin")
                else:
                    self.put_in_place("bin_b_place", "put_in_bin")
                total_cnt += 1

                continue

                # binに入れる
                if total_cnt % 2 == 0:  self.put_in_place("bin_a_place", "put_in_bin")
                else:  self.put_in_place("bin_b_place", "put_in_bin")
                total_cnt += 1

    def execute_task2a(self):
        """
        task2aを実行する(A* Path Planningに基づく経路探索)
        """
        rospy.loginfo("#### start Task 2a (A* Path Planning) ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.goto_name("standby_2a")
        START_X = self.coordinates["positions"]["standby_2a"][0]
        
        self.change_pose("look_at_near_floor")

        # 2. 障害物検出とグリッドマップの生成
        detected_objs = self.get_latest_detection()
        # NOTE: bboxのtf座標取得は検出結果が安定していると仮定し、ここに集中させています
        pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in detected_objs.bboxes]
        
        # コストマップを生成
        cost_map, grid_res_x, grid_res_y = self._create_cost_map(pos_bboxes)
        
        # 3. 最適なウェイポイントのリストをA*で取得
        # path_waypoints: [[x1, y1], [x2, y2], ...]
        path_waypoints = self._find_optimal_path(cost_map, grid_res_x, grid_res_y, START_X) 
        
        if not path_waypoints:
            rospy.logerr("A* path planning failed. Moving to goal directly (RISKY).")
            self.goto_name("go_throw_2a")
            whole_body.move_to_go()
            return

        # 4. 経路に沿って連続移動
        current_x = START_X
        # standby_2a の Y 座標を現在の Y 座標として使用
        current_y = self.coordinates["positions"]["standby_2a"][1] 
        current_yaw = self.coordinates["positions"]["standby_2a"][2]

        for i, target_point in enumerate(path_waypoints):
            target_x, target_y = target_point
            
            # 最終目標点の場合、yaw は go_throw_2a のものを使用
            if i == len(path_waypoints) - 1:
                yaw = self.coordinates["positions"]["go_throw_2a"][2]
            else:
                # 目標方向へのYaw角を計算
                yaw = self._calculate_yaw(current_x, current_y, target_x, target_y)
            
            # goto_pos は [x, y, yaw] のリストを期待
            waypoint = [target_x, target_y, yaw]
            
            rospy.loginfo("Path Step {}: Moving to X={:.2f}, Y={:.2f}, Yaw={:.2f}".format(
                i + 1, target_x, target_y, yaw))
                
            self.goto_pos(waypoint) 
            
            # 現在位置を更新
            current_x, current_y, current_yaw = target_x, target_y, yaw

        # 5. 最終ポーズへ (A*の経路に最終目標点が含まれているため、これで完了)
        whole_body.move_to_go()

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")

        # 命令文を取得
        if self.instruction_list:
            latest_instruction = self.instruction_list[-1]
            rospy.loginfo("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        self.execute_task1()
        self.execute_task2a()
        self.execute_task2b()

    def continue_task1(self):
        end = time.time()
        np.append(self.timestamp, end - self.timestamp[-1])
        if self.task1_ave is None:
            return True
        
        self.task1_ave = self.timestamp[1:].mean()
        if end + self.task1_ave * 2 >= self.task1_timelim:
            return False
        else:
            return True


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
