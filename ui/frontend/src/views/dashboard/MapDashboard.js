import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

const MapDashboard = () => {
  const mapContainerRef = useRef(null); // Map container element
  const mapInstanceRef = useRef(null); // Leaflet map instance
  const markersLayerRef = useRef(null); // Markers layer
  const allMarkersRef = useRef([]); // Store all markers for filtering
  const [roadData, setRoadData] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const roadsAPI = "http://localhost:3001/lane"

  useEffect(() => {
    const fetchRoadData = async () => {
      try {
        const response = await fetch(roadsAPI);
        const data = await response.json();
        setRoadData(data["data"]["lane"]);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching road data:", error);
        setLoading(false);
      }
    };

    fetchRoadData(); 
  }, []);

  useEffect(() => {
    if (roadData.length > 0 && !mapInstanceRef.current) {
      // Load Leaflet's stylesheet
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "https://unpkg.com/leaflet/dist/leaflet.css";
      document.head.appendChild(link);

      // Load Leaflet's JavaScript
      const script = document.createElement("script");
      script.src = "https://unpkg.com/leaflet/dist/leaflet.js";
      script.onload = () => {
        if (mapContainerRef.current && !mapInstanceRef.current) {
          const map = window.L.map(mapContainerRef.current).setView([1.3521, 103.8198], 12);

          window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "&copy; OpenStreetMap contributors",
          }).addTo(map);

          mapInstanceRef.current = map;

          // Create a layer group to hold markers
          const markersLayer = window.L.layerGroup().addTo(map);
          markersLayerRef.current = markersLayer;

          const allMarkers = [];
          const uniqueLocations = [...new Set(roadData.map(roads => roads.road))];
          const roadFilter = document.getElementById("roadFilter");

          uniqueLocations.forEach((road) => {
            const option = document.createElement("option");
            option.value = road;
            option.text = road;
            roadFilter.appendChild(option);
          });

          // Markers for each road
          roadData.forEach((roads) => {
            const marker = window.L.marker([roads.lat, roads.longi])
              .addTo(markersLayer)
              .bindTooltip(`Camera ID: ${roads.camera}<br>Road: ${roads.road}`)
              .on("click", () => {
                navigate(`/dashboard/metrics?id=${roads.camera}`);
              });

            marker.location = roads.road;
            allMarkers.push(marker);
          });

          // Stores markers for filtering, removes all and adds them back if exists in the filter
          allMarkersRef.current = allMarkers; 
          function filterMarkers(selectedLocation) {
            markersLayer.clearLayers();

            allMarkersRef.current.forEach((marker) => {
              if (selectedLocation === "all" || marker.location === selectedLocation) {
                markersLayer.addLayer(marker);
              }
            });
          }

          roadFilter.addEventListener("change", function () {
            const selectedLocation = this.value;
            filterMarkers(selectedLocation);
          });
        }
      };

      document.body.appendChild(script);

      return () => {
        if (mapInstanceRef.current) {
          mapInstanceRef.current.remove();
          mapInstanceRef.current = null;
        }
        document.head.removeChild(link);
        document.body.removeChild(script);
      };
    }
  }, [roadData, navigate]); // Only runs after data is fetched

  // If road data is still not fetched
  if (loading) {
    return <p>Loading map...</p>;
  }

  return (
<div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", width: "100%" }}>
  <div id="filter" style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", padding: "10px", width: "100%", marginBottom: "10px" }}>
    <p style={{ margin: "0 10px 0 0" }}><i className="fa-solid fa-filter"></i> Location:</p>
    <select id="roadFilter" style={{ width: "200px", padding: "5px" }}>
      <option value="all">All Locations</option>
    </select>
  </div>
  <div id="map" ref={mapContainerRef} style={{ height: "350px", width: "100%" }}></div>
</div>


  );
};

export default MapDashboard;