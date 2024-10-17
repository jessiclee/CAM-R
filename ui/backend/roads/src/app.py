import os
import socket
import json
import requests
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import asc
from flask_cors import CORS
import os

os.environ['PYTHONUNBUFFERED'] = '1'

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = \
#         "mysql+mysqlconnector://root:root@localhost:3306" + '/roads'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 100,
#                                             'pool_recycle': 280}

if os.environ.get('db_conn'):
    app.config['SQLALCHEMY_DATABASE_URI'] = \
            os.environ.get('db_conn') + '/roads'
else:
    # app.config['SQLALCHEMY_DATABASE_URI'] = \
    #         'mysql+mysqlconnector://cs302:cs302@localhost:3306/events'
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        'mysql+mysqlconnector://user:root@host.docker.internal:30000/roads'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 100,
                                           'pool_recycle': 280}

db = SQLAlchemy(app)
CORS(app)

# Lane Database
class Lane(db.Model):
    __tablename__ = 'lane'

    camera = db.Column(db.BigInteger, primary_key=True)
    road = db.Column(db.String(100), nullable=False)
    width = db.Column(db.BigInteger, nullable=False)
    height = db.Column(db.BigInteger, nullable=False)
    longi = db.Column(db.Float(precision=10), nullable=False)
    lat = db.Column(db.Float(precision=10), nullable=False)
    lanes = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, camera, road, width, height, longi, lat, lanes):
        self.camera = camera
        self.road = road
        self.height = height
        self.width = width
        self.longi = longi
        self.lat = lat
        self.lanes = lanes

    def to_dict(self):
        return {
            "camera": self.camera,
            "road": self.road,
            "height": self.height,
            "width": self.width,
            "longi": self.longi,
            "lat": self.lat,
            "lanes": self.lanes.decode('utf-8')
        }
        
# Direction Database
class Direction(db.Model):
    __tablename__ = 'direction'

    camera = db.Column(db.BigInteger, db.ForeignKey('lane.camera', ondelete='CASCADE', onupdate='CASCADE'), primary_key=True)
    lane = db.Column(db.BigInteger, primary_key=True)
    direction = db.Column(db.String(10), nullable=False)

    def __init__(self, camera, lane, direction):
        self.camera = camera
        self.lane = lane
        self.direction = direction

    def to_dict(self):
        return {
            "camera": self.camera,
            "lane": self.lane,
            "direction": self.direction
        }

# # Ensure the tables are created
# with app.app_context():
#     db.create_all()  # This creates the tables

@app.route("/health")
def health_check():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    return jsonify(
            {
                "message": "Service is healthy.",
                "service:": "roads",
                "ip_address": local_ip
            }
    ), 200

"""
LANE TABLE REQUESTS
"""
@app.route("/lane")
def get_all():
    lane_list = db.session.scalars(
                    db.select(Lane)
                ).all()
    if len(lane_list) != 0:
        return jsonify(
            {
                "data": {
                    "lane": [lane.to_dict() for lane in lane_list]
                }
            }
        ), 200
    return jsonify(
        {
            "message": "There are no cameras."
        }
    ), 404

@app.route("/lane/<int:cam_id>")
def find_by_id(cam_id):
    camera = db.session.scalars(
                db.select(Lane).
                filter_by(camera=cam_id).
                limit(1)
           ).first()
    if camera:
        return jsonify(
            {
                "data": camera.to_dict()
            }
        ), 200
    return jsonify(
        {
            "message": "Camera not found."
        }
    ), 404

@app.route("/lane", methods=['POST'])
def new_lane():
    try:
        # Ensure the request has the 'camera', 'longi', and 'lat' fields
        if 'camera' not in request.form or 'road' not in request.form or 'longi' not in request.form or 'lat' not in request.form or 'width' not in request.form or 'height' not in request.form:
            return jsonify({'error': 'Camera, road, width, height, longitude, and latitude are required'}), 400

        camera = request.form['camera']
        road = request.form['road']
        width = request.form['width']
        height = request.form['height']
        longi = float(request.form['longi'])
        lat = float(request.form['lat'])

        # Ensure the file is included in the request
        if 'lanes' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        lanes = request.files['lanes']
        # Read the binary data from the file
        lanes = lanes.read()
        
        lane = Lane(camera=camera, road = road, width=width, height=height, longi=longi, lat=lat, lanes=lanes)
        db.session.add(lane)
        db.session.commit()
    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred creating the camera lane information.",
                "error": str(e)
            }
        ), 500

    return jsonify(
        {
            "data": lane.to_dict()
        }
    ), 201

@app.route("/lane/<int:cam_id>", methods=['PATCH'])
def update_lane(cam_id):
    lane = db.session.scalars(
                db.select(Lane).
                with_for_update(of=Lane).
                filter_by(camera=cam_id).
                limit(1)
            ).first()
    if lane is None:
        return jsonify(
            {
                "data": {
                    "camera": cam_id
                },
                "message": "camera not found."
            }
        ), 404

    if "camera" in request.form:
        lane.camera = request.form['camera']
    if "road" in request.form:
        lane.road = request.form['road']
    if "width" in request.form:
        lane.width = request.form['width']
    if "height" in request.form:
        lane.height = request.form['height']
    if "longi" in request.form:
        lane.longi = float(request.form['longi'])
    if "lat" in request.form:
        lane.lat = float(request.form['lat'])

    # Ensure the file is included in the request
    if 'lanes' in request.files:
        lanes = request.files['lanes']
        # Read the binary data from the file
        lane.lanes = lanes.read()

    try:
        db.session.commit()
    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred updating the camera info.",
                "error": str(e)
            }
        ), 500
    return jsonify(
        {
            "data": lane.to_dict()
        }
    )

@app.route("/lane/<int:cam_id>", methods=['DELETE'])
def delete_lane(cam_id):
    camera = db.session.scalars(
                db.select(Lane).
                filter_by(camera=cam_id).
                limit(1)
            ).first()
    if camera is not None:
        try:
            db.session.delete(camera)
            db.session.commit()
        except Exception as e:
            return jsonify(
                {
                    "message": "An error occurred deleting the cam.",
                    "error": str(e)
                }
            ), 500
        return jsonify(
            {
                "data": {
                    "camera": cam_id
                }
            }
        ), 200
    return jsonify(
        {
            "data": {
                "camera": cam_id
            },
            "message": "cam not found."
        }
    ), 404


"""
DIRECTION TABLE REQUESTS
"""
@app.route("/direction")
def get_all_directions():
    direc_list = db.session.scalars(
                    db.select(Direction)
                ).all()
    if len(direc_list) != 0:
        return jsonify(
            {
                "data": {
                    "direction": [direc.to_dict() for direc in direc_list]
                }
            }
        ), 200
    return jsonify(
        {
            "message": "There are no cameras."
        }
    ), 404

@app.route("/direction/<int:cam_id>/<int:lane_id>")
def find_by_camidlaneid(cam_id, lane_id):
    direc = db.session.scalars(
                db.select(Direction).
                filter_by(camera=cam_id, lane=lane_id).
                limit(1)
            ).first()
    if direc:
        return jsonify(
            {
                "data": direc.to_dict()
            }
        ), 200
    return jsonify(
        {
            "message": "Camera not found."
        }
    ), 404

@app.route("/direction/<int:cam_id>")
def find_by_camid(cam_id):
    direc_list = list(db.session.scalars(
                db.select(Direction).
                filter_by(camera=cam_id).
                order_by(asc(Direction.lane))
            ))
    direction_list = [item.direction for item in direc_list]
    if len(direc_list) != 0:
        return jsonify(
            {
                "data": direction_list
            }
        ), 200
    return jsonify(
        {
            "message": "Camera not found."
        }
    ), 404

@app.route("/direction", methods=['POST'])
def new_direction():
    try:
        # Ensure the request has the 'camera', 'longi', and 'lat' fields
        if 'camera' not in request.form or 'lane' not in request.form or 'direction' not in request.form:
            return jsonify({'error': 'Camera, lane and direction are required'}), 400

        camera = request.form['camera']
        lane = request.form['lane']
        direction = request.form['direction']
        direc = Direction(camera=camera, lane = lane, direction=direction)
        db.session.add(direc)
        db.session.commit()
    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred creating the camera lane information.",
                "error": str(e)
            }
        ), 500

    return jsonify(
        {
            "data": direc.to_dict()
        }
    ), 201

@app.route("/direction/<int:cam_id>/<int:lane_id>", methods=['PATCH'])
def update_direction(cam_id, lane_id):
    direc = db.session.scalars(
                db.select(Direction).
                with_for_update(of=Direction).
                filter_by(camera=cam_id, lane=lane_id).
                limit(1)
            ).first()
    if direc is None:
        return jsonify(
            {
                "data": {
                    "camera": cam_id,
                    "lane": lane_id
                },
                "message": "lane direction not found."
            }
        ), 404

    if "camera" in request.form:
        direc.camera = request.form['camera']
    if "lane" in request.form:
        direc.lane = request.form['lane']
    if "direction" in request.form:
        direc.direction = request.form['direction']

    try:
        db.session.commit()
    except Exception as e:
        return jsonify(
            {
                "message": "An error occurred updating the direction info.",
                "error": str(e)
            }
        ), 500
    return jsonify(
        {
            "data": direc.to_dict()
        }
    )

@app.route("/direction/<int:cam_id>/<int:lane_id>", methods=['DELETE'])
def delete_direction(cam_id, lane_id):
    direc = db.session.scalars(
                db.select(Direction).
                filter_by(camera=cam_id, lane = lane_id).
                limit(1)
            ).first()
    if direc is not None:
        try:
            db.session.delete(direc)
            db.session.commit()
        except Exception as e:
            return jsonify(
                {
                    "message": "An error occurred deleting the lane direction.",
                    "error": str(e)
                }
            ), 500
        return jsonify(
            {
                "data": {
                    "camera": cam_id,
                    "lane": lane_id
                }
            }
        ), 200
    return jsonify(
        {
            "data": {
                "camera": cam_id,
                "lane": lane_id
            },
            "message": "lane direction not found."
        }
    ), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)
