import React, { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom"; // Updated to useNavigate

const MapDashboard = () => {
  const mapContainerRef = useRef(null); // Ref for the map container element
  const mapInstanceRef = useRef(null); // Ref to store the Leaflet map instance
  const markersLayerRef = useRef(null); // Ref to store the markers layer
  const allMarkersRef = useRef([]); // Ref to store all markers for filtering
  const navigate = useNavigate(); // Use useNavigate instead of useHistory

  useEffect(() => {
    // Dynamically load Leaflet's stylesheet
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://unpkg.com/leaflet/dist/leaflet.css";
    document.head.appendChild(link);

    // Dynamically load Leaflet's JavaScript
    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet/dist/leaflet.js";
    script.onload = () => {
      // Initialize the map once Leaflet is loaded
      if (mapContainerRef.current && !mapInstanceRef.current) {
        const map = window.L.map(mapContainerRef.current).setView([1.3521, 103.8198], 12);

        // Add OpenStreetMap tile layer
        window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "&copy; OpenStreetMap contributors",
        }).addTo(map);

        mapInstanceRef.current = map; // Store the map instance

        // mocked data
        const allRoadsJson = {
          "data": {
            "lane": [
              {
                "camera": 1001,
                "height": 240,
                "lanes": "92.000000,2.000000 193.000000,239.000000\r\n74.000000,0.000000 118.000000,250.000000\r\n164.000000,2.000000 320.000000,154.000000\r\n181.000000,1.000000 319.000000,103.000000\r\n205.000000,2.000000 319.000000,77.000000\r\n",
                "lat": 1.29531,
                "longi": 103.871,
                "road": "EAST COAST PARKWAY",
                "width": 320
              },
              {
                "camera": 1002,
                "height": 240,
                "lanes": "45.000000,71.000000 36.000000,85.000000 43.000000,111.000000 102.000000,237.000000\r\n0.000000,0.000000 43.000000,250.000000\r\n101.000000,76.000000 112.000000,114.000000 167.000000,179.000000\r\n",
                "lat": 1.31954,
                "longi": 103.879,
                "road": "PAN ISLAND EXPRESSWAY",
                "width": 320
              },
              {
                "camera": 1003,
                "height": 240,
                "lanes": "3.000000,150.000000 44.000000,87.000000\r\n4.000000,122.000000 35.000000,82.000000\r\n50.000000,0.000000 92.000000,250.000000\r\n124.000000,99.000000 165.000000,141.000000 275.000000,239.000000\r\n28.000000,0.000000 226.000000,250.000000\r\n320.000000,140.000000 205.000000,98.000000 147.000000,76.000000\r\n317.000000,150.000000 227.000000,114.000000 144.000000,79.000000\r\n318.000000,162.000000 226.000000,124.000000 175.000000,100.000000 141.000000,81.000000\r\n318.000000,182.000000 212.000000,127.000000 137.000000,86.000000\r\n",
                "lat": 1.32396,
                "longi": 103.873,
                "road": "PAN ISLAND EXPRESSWAY",
                "width": 320
              }
            ]
          }
        }

        // const roadData = allRoadsJson["data"]["lane"]

        // all just to see and check
        const roadData = [
          { "camera": 1001, "road": "East Coast Parkway", "lat": 1.29531332, "longi": 103.871146 },
          { "camera": 1002, "road": "Pan Island Expressway", "lat": 1.319541067, "longi": 103.8785627 },
          { "camera": 1003, "road": "Pan Island Expressway", "lat": 1.323957439, "longi": 103.8728576 },
          { "camera": 1004, "road": "Kallang Paya Lebar Expressway", "lat": 1.319535712, "longi": 103.8750668 },
          { "camera": 1005, "road": "Kallang Paya Lebar Expressway", "lat": 1.363519886, "longi": 103.905394 },
          { "camera": 1006, "road": "Kallang Paya Lebar Expressway", "lat": 1.357098686, "longi": 103.902042 },
          { "camera": 1111, "road": "Tampines Expressway", "lat": 1.365434, "longi": 103.953997 },
          { "camera": 1112, "road": "Tampines Expressway", "lat": 1.3605, "longi": 103.961412 },
          { "camera": 1113, "road": "East Coast Parkway", "lat": 1.317036, "longi": 103.988598 },
          { "camera": 1501, "road": "Marina Coastal Expressway", "lat": 1.274143944, "longi": 103.8513168 },
          { "camera": 1502, "road": "Marina Coastal Expressway", "lat": 1.271350907, "longi": 103.8618284 },
          { "camera": 1503, "road": "Marina Coastal Expressway", "lat": 1.270664087, "longi": 103.8569779 },
          { "camera": 1504, "road": "Kallang Paya Lebar Expressway", "lat": 1.294098914, "longi": 103.8760562 },
          { "camera": 1505, "road": "Marina Coastal Expressway", "lat": 1.275297715, "longi": 103.8663904 },
          { "camera": 1701, "road": "Central Expressway", "lat": 1.323604823, "longi": 103.8587802 },
          { "camera": 1702, "road": "Central Expressway", "lat": 1.34355015, "longi": 103.8601984 },
          { "camera": 1703, "road": "Central Expressway", "lat": 1.328147222, "longi": 103.8622033 },
          { "camera": 1704, "road": "Central Expressway", "lat": 1.285693989, "longi": 103.8375245 },
          { "camera": 1705, "road": "Central Expressway", "lat": 1.375925022, "longi": 103.8587986 },
          { "camera": 1706, "road": "Central Expressway", "lat": 1.38861, "longi": 103.85806 },
          { "camera": 1707, "road": "Central Expressway", "lat": 1.280365843, "longi": 103.8304511 },
          { "camera": 1709, "road": "Central Expressway", "lat": 1.313842317, "longi": 103.845603 },
          { "camera": 1711, "road": "Central Expressway", "lat": 1.35296, "longi": 103.85719 },
          { "camera": 2701, "road": "Woodlands Checkpoint", "lat": 1.447023728, "longi": 103.7716543 },
          { "camera": 2702, "road": "Woodlands Checkpoint", "lat": 1.445554109, "longi": 103.7683397 },
          { "camera": 2703, "road": "Bukit Timah Expressway", "lat": 1.350477908, "longi": 103.7910336 },
          { "camera": 2704, "road": "Bukit Timah Expressway", "lat": 1.429588536, "longi": 103.769311 },
          { "camera": 2705, "road": "Bukit Timah Expressway", "lat": 1.36728572, "longi": 103.7794698 },
          { "camera": 2706, "road": "Bukit Timah Expressway", "lat": 1.414142, "longi": 103.771168 },
          { "camera": 2707, "road": "Bukit Timah Expressway", "lat": 1.3983, "longi": 103.774247 },
          { "camera": 2708, "road": "Bukit Timah Expressway", "lat": 1.3865, "longi": 103.7747 },
          { "camera": 3702, "road": "East Coast Parkway", "lat": 1.33831, "longi": 103.98032 },
          { "camera": 3704, "road": "Kallang Paya Lebar Expressway", "lat": 1.295855016, "longi": 103.8803147 },
          { "camera": 3705, "road": "East Coast Parkway", "lat": 1.32743, "longi": 103.97383 },
          { "camera": 3793, "road": "East Coast Parkway", "lat": 1.309330837, "longi": 103.9350504 },
          { "camera": 3795, "road": "East Coast Parkway", "lat": 1.301451452, "longi": 103.9105963 },
          { "camera": 3796, "road": "East Coast Parkway", "lat": 1.297512569, "longi": 103.8983019 },
          { "camera": 3797, "road": "East Coast Parkway", "lat": 1.295657333, "longi": 103.885283 },
          { "camera": 3798, "road": "East Coast Parkway", "lat": 1.29158484, "longi": 103.8615987 },
          { "camera": 4701, "road": "Ayer Rajah Expressway", "lat": 1.2871, "longi": 103.79633 },
          { "camera": 4702, "road": "Ayer Rajah Expressway", "lat": 1.27237, "longi": 103.8324 },
          { "camera": 4703, "road": "Tuas Checkpoint", "lat": 1.348697862, "longi": 103.6350413 },
          { "camera": 4704, "road": "Ayer Rajah Expressway", "lat": 1.27877, "longi": 103.82375 },
          { "camera": 4705, "road": "Ayer Rajah Expressway", "lat": 1.32618, "longi": 103.73028 },
          { "camera": 4706, "road": "Ayer Rajah Expressway", "lat": 1.29792, "longi": 103.78205 },
          { "camera": 4707, "road": "Ayer Rajah Expressway", "lat": 1.333446481, "longi": 103.6527008 },
          { "camera": 4708, "road": "Ayer Rajah Expressway", "lat": 1.29939, "longi": 103.7799 },
          { "camera": 4709, "road": "Ayer Rajah Expressway", "lat": 1.312019, "longi": 103.763002 },
          { "camera": 4710, "road": "Ayer Rajah Expressway", "lat": 1.32153, "longi": 103.75273 },
          { "camera": 4712, "road": "Ayer Rajah Expressway", "lat": 1.341244001, "longi": 103.6439134 },
          { "camera": 4713, "road": "Tuas Checkpoint", "lat": 1.347645829, "longi": 103.6366955 },
          { "camera": 4714, "road": "Ayer Rajah Expressway", "lat": 1.31023, "longi": 103.76438 },
          { "camera": 4716, "road": "Ayer Rajah Expressway", "lat": 1.32227, "longi": 103.67453 },
          { "camera": 4798, "road": "Ayer Rajah Expressway", "lat": 1.259999997, "longi": 103.8236111 },
          { "camera": 4799, "road": "Ayer Rajah Expressway", "lat": 1.260277774, "longi": 103.8238889 },
          { "camera": 5794, "road": "Pan Island Expressway", "lat": 1.3309693, "longi": 103.9168616 },
          { "camera": 5795, "road": "Pan Island Expressway", "lat": 1.326024822, "longi": 103.905625 },
          { "camera": 5797, "road": "Pan Island Expressway", "lat": 1.322875288, "longi": 103.8910793 },
          { "camera": 5798, "road": "Kallang Paya Lebar Expressway", "lat": 1.320360781, "longi": 103.8771741 },
          { "camera": 5799, "road": "Pan Island Expressway", "lat": 1.328171608, "longi": 103.8685191 },
          { "camera": 6701, "road": "Pan Island Expressway", "lat": 1.329334, "longi": 103.858222 },
          { "camera": 6703, "road": "Pan Island Expressway", "lat": 1.328899, "longi": 103.84121 },
          { "camera": 6704, "road": "Pan Island Expressway", "lat": 1.326574036, "longi": 103.8268573 },
          { "camera": 6705, "road": "Pan Island Expressway", "lat": 1.332124, "longi": 103.81768 },
          { "camera": 6706, "road": "Pan Island Expressway", "lat": 1.349428893, "longi": 103.7952799 },
          { "camera": 6708, "road": "Pan Island Expressway", "lat": 1.345996, "longi": 103.69016 },
          { "camera": 6710, "road": "Pan Island Expressway", "lat": 1.344205, "longi": 103.78577 },
          { "camera": 6711, "road": "Pan Island Expressway", "lat": 1.33771, "longi": 103.977827 },
          { "camera": 6712, "road": "Pan Island Expressway", "lat": 1.332691, "longi": 103.770278 },
          { "camera": 6713, "road": "Pan Island Expressway", "lat": 1.340298, "longi": 103.945652 },
          { "camera": 6714, "road": "Pan Island Expressway", "lat": 1.361742, "longi": 103.703341 },
          { "camera": 6715, "road": "Pan Island Expressway", "lat": 1.356299, "longi": 103.716071 },
          { "camera": 6716, "road": "Ayer Rajah Expressway", "lat": 1.322893, "longi": 103.6635051 },
          { "camera": 7791, "road": "Tampines Expressway", "lat": 1.354245, "longi": 103.963782 },
          { "camera": 7793, "road": "Tampines Expressway", "lat": 1.37704704, "longi": 103.9294698 },
          { "camera": 7794, "road": "Tampines Expressway", "lat": 1.37988658, "longi": 103.9200917 },
          { "camera": 7795, "road": "Tampines Expressway", "lat": 1.38432741, "longi": 103.915857 },
          { "camera": 7796, "road": "Tampines Expressway", "lat": 1.39559294, "longi": 103.9051571 },
          { "camera": 7797, "road": "Seletar Expressway", "lat": 1.40002575, "longi": 103.8570253 },
          { "camera": 7798, "road": "Seletar Expressway", "lat": 1.39748842, "longi": 103.8540047 },
          { "camera": 8701, "road": "Kranji Expressway", "lat": 1.38647, "longi": 103.74143 },
          { "camera": 8702, "road": "Kranji Expressway", "lat": 1.39059, "longi": 103.7717 },
          { "camera": 8704, "road": "Kranji Expressway", "lat": 1.3899, "longi": 103.74843 },
          { "camera": 8706, "road": "Kranji Expressway", "lat": 1.3664, "longi": 103.70899 },
          { "camera": 9701, "road": "Seletar Expressway", "lat": 1.39466333, "longi": 103.834746 },
          { "camera": 9702, "road": "Seletar Expressway", "lat": 1.39474081, "longi": 103.8179709 },
          { "camera": 9703, "road": "Seletar Expressway", "lat": 1.422857, "longi": 103.773005 },
          { "camera": 9704, "road": "Seletar Expressway", "lat": 1.42214311, "longi": 103.7954206 },
          { "camera": 9705, "road": "Seletar Expressway", "lat": 1.42627712, "longi": 103.7871664 },
          { "camera": 9706, "road": "Seletar Expressway", "lat": 1.41270056, "longi": 103.8064271 }
        ];

        // Create a layer group to hold markers
        const markersLayer = window.L.layerGroup().addTo(map);
        markersLayerRef.current = markersLayer;

        // Store all markers for filtering
        const allMarkers = [];

        // Get unique locations from the roadData using a Set to avoid duplicates
        const uniqueLocations = [...new Set(roadData.map(roads => roads.road))];

        // Populate the dropdown with unique road names
        const roadFilter = document.getElementById("roadFilter");

        uniqueLocations.forEach((road) => {
          const option = document.createElement("option");
          option.value = road;
          option.text = road;
          roadFilter.appendChild(option);
        });

        // Create markers for each road entry
        roadData.forEach((roads) => {
          const marker = window.L.marker([roads.lat, roads.longi])
            .addTo(markersLayer)
            .bindTooltip(`Camera ID: ${roads.camera}<br>Location: ${roads.road}`)
            .on("click", () => {
              navigate(`/dashboard/metrics?id=${roads.camera}`); // Use useNavigate to pass the ID in the URL
            });

          marker.location = roads.road; // Add custom property to marker for filtering
          allMarkers.push(marker);
        });

        allMarkersRef.current = allMarkers; // Store all markers for filtering

        // Filter function to show only selected road markers
        function filterMarkers(selectedLocation) {
          markersLayer.clearLayers(); // Clear all markers

          allMarkersRef.current.forEach((marker) => {
            if (selectedLocation === "all" || marker.location === selectedLocation) {
              markersLayer.addLayer(marker); // Add marker back if it matches the filter
            }
          });
        }

        // Event listener for dropdown change
        roadFilter.addEventListener("change", function () {
          const selectedLocation = this.value;
          filterMarkers(selectedLocation); // Call the filter function
        });
      }
    };

    document.body.appendChild(script);

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove(); // Remove the map instance on unmount
        mapInstanceRef.current = null;
      }
      document.head.removeChild(link);
      document.body.removeChild(script);
    };
  }, [navigate]);

  return (
    <div style={{ display: "flex" }}>
      <div id="map" ref={mapContainerRef} style={{ height: "550px", width: "85%" }}></div>
      <div id="filter" style={{ padding: "10px", width: "20%" }}>
        <h4><i class="fa-solid fa-filter"></i>Filter by Location</h4>
        <select id="roadFilter" style={{ width: "100%", padding: "5px" }}>
          <option value="all">All Locations</option>
        </select>
      </div>
    </div>
  );
};

export default MapDashboard;