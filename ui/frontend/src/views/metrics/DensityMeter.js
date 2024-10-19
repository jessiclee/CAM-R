import React from "react";
import ReactSpeedometer from "react-d3-speedometer";

const DensityMeter = ({ level }) => {
    // Map the level to a numeric value for the pointer
    const getLevelValue = (level) => {
        switch (level) {
            case "Low":
                return 18;
            case "Medium":
                return 50;
            case "High":
                return 80;
            default:
                return 0;
        }
    };

    return (
        // <div>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
            <ReactSpeedometer
                value={getLevelValue(level)}
                minValue={0}
                maxValue={100}
                segments={3} // Divides the gauge into 3 sections
                segmentColors={["#00FF00", "#FFDD00", "#FF0000"]} // Green, Yellow, Red
                needleColor="black"
                startColor="green"
                endColor="red"
                currentValueText="" // Hides the default current value text
                valueTextFontSize="0" // Hides the value inside the gauge
                labelFontSize="0" // Hides all numeric labels (0, 50, 100)
                height={160}
            />
            <h3>{level}</h3>
        </div>

    );
};

export default DensityMeter;
