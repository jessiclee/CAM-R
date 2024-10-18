-- Lane Table
CREATE TABLE lane (
    camera BIGINT PRIMARY KEY,
    road VARCHAR(100) NOT NULL,
    width BIGINT NOT NULL,
    height BIGINT NOT NULL,
    longi FLOAT(10) NOT NULL,
    lat FLOAT(10) NOT NULL,
    lanes LONGBLOB NOT NULL
);

-- Direction Table
CREATE TABLE direction (
    camera BIGINT,
    lane BIGINT,
    direction VARCHAR(10) NOT NULL,
    PRIMARY KEY (camera, lane),
    FOREIGN KEY (camera) REFERENCES lane(camera) ON DELETE CASCADE ON UPDATE CASCADE
);