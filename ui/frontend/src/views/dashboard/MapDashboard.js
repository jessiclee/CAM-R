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

        // Road data
        const roadData = [
          { id: 1001, location: "East Coast Parkway", lat: 1.29531332, lng: 103.871146 },
          { id: 1002, location: "Pan Island Expressway", lat: 1.319541067, lng: 103.8785627 },
          { id: 1003, location: "Pan Island Expressway", lat: 1.323957439, lng: 103.8728576 },
          { id: 1004, location: "Kallang Paya Lebar Expressway", lat: 1.319535712, lng: 103.8750668 },
          { id: 1005, location: "Kallang Paya Lebar Expressway", lat: 1.363519886, lng: 103.905394 },
          { id: 1006, location: "Kallang Paya Lebar Expressway", lat: 1.357098686, lng: 103.902042 },
          { id: 1111, location: "Tampines Expressway", lat: 1.365434, lng: 103.953997 },
          { id: 1112, location: "Tampines Expressway", lat: 1.3605, lng: 103.961412 },
          { id: 1113, location: "East Coast Parkway", lat: 1.317036, lng: 103.988598 },
          { id: 1501, location: "Marina Coastal Expressway", lat: 1.274143944, lng: 103.8513168 },
          { id: 1502, location: "Marina Coastal Expressway", lat: 1.271350907, lng: 103.8618284 },
          { id: 1503, location: "Marina Coastal Expressway", lat: 1.270664087, lng: 103.8569779 },
          { id: 1504, location: "Kallang Paya Lebar Expressway", lat: 1.294098914, lng: 103.8760562 },
          { id: 1505, location: "Marina Coastal Expressway", lat: 1.275297715, lng: 103.8663904 },
          { id: 1701, location: "Central Expressway", lat: 1.323604823, lng: 103.8587802 },
          { id: 1702, location: "Central Expressway", lat: 1.34355015, lng: 103.8601984 },
          { id: 1703, location: "Central Expressway", lat: 1.328147222, lng: 103.8622033 },
          { id: 1704, location: "Central Expressway", lat: 1.285693989, lng: 103.8375245 },
          { id: 1705, location: "Central Expressway", lat: 1.375925022, lng: 103.8587986 },
          { id: 1706, location: "Central Expressway", lat: 1.38861, lng: 103.85806 },
          { id: 1707, location: "Central Expressway", lat: 1.280365843, lng: 103.8304511 },
          { id: 1709, location: "Central Expressway", lat: 1.313842317, lng: 103.845603 },
          { id: 1711, location: "Central Expressway", lat: 1.35296, lng: 103.85719 },
          { id: 2701, location: "Woodlands Checkpoint", lat: 1.447023728, lng: 103.7716543 },
          { id: 2702, location: "Woodlands Checkpoint", lat: 1.445554109, lng: 103.7683397 },
          { id: 2703, location: "Bukit Timah Expressway", lat: 1.350477908, lng: 103.7910336 },
          { id: 2704, location: "Bukit Timah Expressway", lat: 1.429588536, lng: 103.769311 },
          { id: 2705, location: "Bukit Timah Expressway", lat: 1.36728572, lng: 103.7794698 },
          { id: 2706, location: "Bukit Timah Expressway", lat: 1.414142, lng: 103.771168 },
          { id: 2707, location: "Bukit Timah Expressway", lat: 1.3983, lng: 103.774247 },
          { id: 2708, location: "Bukit Timah Expressway", lat: 1.3865, lng: 103.7747 },
          { id: 3702, location: "East Coast Parkway", lat: 1.33831, lng: 103.98032 },
          { id: 3704, location: "Kallang Paya Lebar Expressway", lat: 1.295855016, lng: 103.8803147 },
          { id: 3705, location: "East Coast Parkway", lat: 1.32743, lng: 103.97383 },
          { id: 3793, location: "East Coast Parkway", lat: 1.309330837, lng: 103.9350504 },
          { id: 3795, location: "East Coast Parkway", lat: 1.301451452, lng: 103.9105963 },
          { id: 3796, location: "East Coast Parkway", lat: 1.297512569, lng: 103.8983019 },
          { id: 3797, location: "East Coast Parkway", lat: 1.295657333, lng: 103.885283 },
          { id: 3798, location: "East Coast Parkway", lat: 1.29158484, lng: 103.8615987 },
          { id: 4701, location: "Ayer Rajah Expressway", lat: 1.2871, lng: 103.79633 },
          { id: 4702, location: "Ayer Rajah Expressway", lat: 1.27237, lng: 103.8324 },
          { id: 4703, location: "Tuas Checkpoint", lat: 1.348697862, lng: 103.6350413 },
          { id: 4704, location: "Ayer Rajah Expressway", lat: 1.27877, lng: 103.82375 },
          { id: 4705, location: "Ayer Rajah Expressway", lat: 1.32618, lng: 103.73028 },
          { id: 4706, location: "Ayer Rajah Expressway", lat: 1.29792, lng: 103.78205 },
          { id: 4707, location: "Ayer Rajah Expressway", lat: 1.333446481, lng: 103.6527008 },
          { id: 4708, location: "Ayer Rajah Expressway", lat: 1.29939, lng: 103.7799 },
          { id: 4709, location: "Ayer Rajah Expressway", lat: 1.312019, lng: 103.763002 },
          { id: 4710, location: "Ayer Rajah Expressway", lat: 1.32153, lng: 103.75273 },
          { id: 4712, location: "Ayer Rajah Expressway", lat: 1.341244001, lng: 103.6439134 },
          { id: 4713, location: "Tuas Checkpoint", lat: 1.347645829, lng: 103.6366955 },
          { id: 4714, location: "Ayer Rajah Expressway", lat: 1.31023, lng: 103.76438 },
          { id: 4716, location: "Ayer Rajah Expressway", lat: 1.32227, lng: 103.67453 },
          { id: 4798, location: "Ayer Rajah Expressway", lat: 1.259999997, lng: 103.8236111 },
          { id: 4799, location: "Ayer Rajah Expressway", lat: 1.260277774, lng: 103.8238889 },
          { id: 5794, location: "Pan Island Expressway", lat: 1.3309693, lng: 103.9168616 },
          { id: 5795, location: "Pan Island Expressway", lat: 1.326024822, lng: 103.905625 },
          { id: 5797, location: "Pan Island Expressway", lat: 1.322875288, lng: 103.8910793 },
          { id: 5798, location: "Kallang Paya Lebar Expressway", lat: 1.320360781, lng: 103.8771741 },
          { id: 5799, location: "Pan Island Expressway", lat: 1.328171608, lng: 103.8685191 },
          { id: 6701, location: "Pan Island Expressway", lat: 1.329334, lng: 103.858222 },
          { id: 6703, location: "Pan Island Expressway", lat: 1.328899, lng: 103.84121 },
          { id: 6704, location: "Pan Island Expressway", lat: 1.326574036, lng: 103.8268573 },
          { id: 6705, location: "Pan Island Expressway", lat: 1.332124, lng: 103.81768 },
          { id: 6706, location: "Pan Island Expressway", lat: 1.349428893, lng: 103.7952799 },
          { id: 6708, location: "Pan Island Expressway", lat: 1.345996, lng: 103.69016 },
          { id: 6710, location: "Pan Island Expressway", lat: 1.344205, lng: 103.78577 },
          { id: 6711, location: "Pan Island Expressway", lat: 1.33771, lng: 103.977827 },
          { id: 6712, location: "Pan Island Expressway", lat: 1.332691, lng: 103.770278 },
          { id: 6713, location: "Pan Island Expressway", lat: 1.340298, lng: 103.945652 },
          { id: 6714, location: "Pan Island Expressway", lat: 1.361742, lng: 103.703341 },
          { id: 6715, location: "Pan Island Expressway", lat: 1.356299, lng: 103.716071 },
          { id: 6716, location: "Ayer Rajah Expressway", lat: 1.322893, lng: 103.6635051 },
          { id: 7791, location: "Tampines Expressway", lat: 1.354245, lng: 103.963782 },
          { id: 7793, location: "Tampines Expressway", lat: 1.37704704, lng: 103.9294698 },
          { id: 7794, location: "Tampines Expressway", lat: 1.37988658, lng: 103.9200917 },
          { id: 7795, location: "Tampines Expressway", lat: 1.38432741, lng: 103.915857 },
          { id: 7796, location: "Tampines Expressway", lat: 1.39559294, lng: 103.9051571 },
          { id: 7797, location: "Seletar Expressway", lat: 1.40002575, lng: 103.8570253 },
          { id: 7798, location: "Seletar Expressway", lat: 1.39748842, lng: 103.8540047 },
          { id: 8701, location: "Kranji Expressway", lat: 1.38647, lng: 103.74143 },
          { id: 8702, location: "Kranji Expressway", lat: 1.39059, lng: 103.7717 },
          { id: 8704, location: "Kranji Expressway", lat: 1.3899, lng: 103.74843 },
          { id: 8706, location: "Kranji Expressway", lat: 1.3664, lng: 103.70899 },
          { id: 9701, location: "Seletar Expressway", lat: 1.39466333, lng: 103.834746 },
          { id: 9702, location: "Seletar Expressway", lat: 1.39474081, lng: 103.8179709 },
          { id: 9703, location: "Seletar Expressway", lat: 1.422857, lng: 103.773005 },
          { id: 9704, location: "Seletar Expressway", lat: 1.42214311, lng: 103.7954206 },
          { id: 9705, location: "Seletar Expressway", lat: 1.42627712, lng: 103.7871664 },
          { id: 9706, location: "Seletar Expressway", lat: 1.41270056, lng: 103.8064271 }
        ];

        // Create a layer group to hold markers
        const markersLayer = window.L.layerGroup().addTo(map);
        markersLayerRef.current = markersLayer;

        // Store all markers for filtering
        const allMarkers = [];

        // Get unique locations from the roadData using a Set to avoid duplicates
        const uniqueLocations = [...new Set(roadData.map(road => road.location))];

        // Populate the dropdown with unique road names
        const roadFilter = document.getElementById("roadFilter");

        uniqueLocations.forEach((location) => {
          const option = document.createElement("option");
          option.value = location;
          option.text = location;
          roadFilter.appendChild(option);
        });

        // Create markers for each road entry
        roadData.forEach((road) => {
          const marker = window.L.marker([road.lat, road.lng])
            .addTo(markersLayer)
            .bindTooltip(`Camera ID: ${road.id}<br>Location: ${road.location}`)
            .on("click", () => {
              navigate(`/dashboard/metrics?id=${road.id}`); // Use useNavigate to pass the ID in the URL
            });

          marker.location = road.location; // Add custom property to marker for filtering
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
        <select id="roadFilter" style={{ width: "100%", padding: "5px"}}>
          <option value="all">All Locations</option>
        </select>
      </div>
    </div>
  );
};

export default MapDashboard;