import React, { useState } from "react";

const QueueTable = () => {
  // hardcode first
  const initialData = [
    { laneID: 1, queue_length: 30 },
    { laneID: 2, queue_length: 45 },
    { laneID: 3, queue_length: 50 },
    { laneID: 4, queue_length: 25 },
    { laneID: 5, queue_length: 60 },
    { laneID: 6, queue_length: 35 },
    { laneID: 7, queue_length: 40 },
    { laneID: 8, queue_length: 55 },
  ];

  // States
  const [data, setData] = useState(initialData); // Data to display
  const [currentPage, setCurrentPage] = useState(1); // Current page
  const [rowsPerPage] = useState(4); // Rows per page
  const [sortDirection, setSortDirection] = useState(false); // Sorting direction

  // Calculate pagination
  const totalPages = Math.ceil(data.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const currentData = data.slice(startIndex, startIndex + rowsPerPage);

  // Function to handle sorting
  const sortTable = (columnKey) => {
    const sortedData = [...data].sort((a, b) => {
      const valueA = a[columnKey];
      const valueB = b[columnKey];
      if (valueA < valueB) return sortDirection ? 1 : -1;
      if (valueA > valueB) return sortDirection ? -1 : 1;
      return 0;
    });
    setSortDirection(!sortDirection); // Toggle sort direction
    setData(sortedData); // Update the sorted data
  };

  // Function to change page
  const handlePageChange = (pageNum) => {
    setCurrentPage(pageNum);
  };

  return (
    <div>
      {/* Table */}
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th
              onClick={() => sortTable("laneID")}
              style={{ cursor: "pointer", padding: "10px", borderBottom: "1px solid #ddd" }}
            >
              Lane ID
            </th>
            <th
              onClick={() => sortTable("queue_length")}
              style={{ cursor: "pointer", padding: "10px", borderBottom: "1px solid #ddd" }}
            >
              Queue Length
            </th>
          </tr>
        </thead>
        <tbody>
          {currentData.map((row, index) => (
            <tr key={index}>
              <td style={{ padding: "10px", borderBottom: "1px solid #ddd" }}>{row.laneID}</td>
              <td style={{ padding: "10px", borderBottom: "1px solid #ddd" }}>{row.queue_length}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      <div style={{ textAlign: "right", margin: "20px 0" }}>
        {Array.from({ length: totalPages }, (_, index) => (
          <button
            key={index}
            onClick={() => handlePageChange(index + 1)}
            style={{
              padding: "10px 15px",
              margin: "0 5px",
              backgroundColor: index + 1 === currentPage ? "#5856DC" : "#ccc",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            {index + 1}
          </button>
        ))}
      </div>
    </div>
  );
};

export default QueueTable;
