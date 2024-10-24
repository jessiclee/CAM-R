import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import DensityMeter from './DensityMeter';
import QueueTable from './QueueTable';
import predictImage from '../../assets/images/queue.jpg'
import {
    CAvatar,
    CButton,
    CButtonGroup,
    CCard,
    CCardBody,
    CCardFooter,
    CCardHeader,
    CCol,
    CProgress,
    CRow,
    CTable,
    CTableBody,
    CTableDataCell,
    CTableHead,
    CTableHeaderCell,
    CTableRow,
    CTooltip
} from '@coreui/react'


const Metrics = () => {
    // Getting specific camera ID information
    const location = useLocation();
    const searchParams = new URLSearchParams(location.search);
    const id = searchParams.get('id');

    // Navigate befor computing is done
    const navigate = useNavigate();
    useEffect(() => {
        const numericID = Number(id);
        if (isNaN(numericID) || numericID <= 0) {
            console.log('Redirecting due to invalid number');
            setTimeout(() => {
                navigate('/');
            }, 200);
            return;
        }
    }, []);

    // Timestamp for Image
    const [timestamp, setTimestamp] = useState('');

    // Base URL
    const metrics_base_url = "http://localhost:3002/"
    
    // Queue Data
    const [queueData, setQueueData] = useState([]);
    const [currQueue, setCurrQueue] = useState([]);
    const postQueue = async () => {
        const queue_url = metrics_base_url + "get_queue";
        const queue_data = { id: [id] };

        try {
            const response = await fetch(queue_url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(queue_data),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const result = await response.json();
            console.log(result)
            setQueueData(result['queues']);
        } catch (error) {
            console.error('Error making the POST request to the get_queue service:', error);
        }
    };

    // Density Data
    const [density_res, set_density_res] = useState(null);
    const [currDensity, setCurrDensity] = useState('N.A.');
    const postDensity = async () => {
        const density_url = metrics_base_url + "density";
        const density_data = { id: [id] };

        try {
            const response = await fetch(density_url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(density_data),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }

            const result = await response.json();
            set_density_res(result);
        } catch (error) {
            console.error('Error making the POST request to the density service:', error);
        }
    };

    useEffect(() => {

        const localTimestamp = new Date().toLocaleString();
        setTimestamp(localTimestamp);

        // Send API requests
        postDensity();
        postQueue();

    }, []);

    // Update Density
    useEffect(() => {
        if (density_res && density_res[id]) {
            setCurrDensity(density_res[id]);
        }
    }, [density_res, id]);

    // Update Queue
    useEffect(() => {
        let transformedQueueData = []

        if (queueData && queueData[id]) {
            // Transform the API data into the format for QueueTable
            transformedQueueData = Object.entries(queueData[id]).map(([laneID, queueLength]) => ({
                laneID: Number(laneID),
                queue_length: queueLength,
            }));

            setCurrQueue(transformedQueueData);
        }
    }, [queueData, id]);

    return (
        <>
            <h1>Metrics: {id}</h1>
            <CRow>
                <CCol xs={5}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Current View @ {timestamp}</strong>
                            <CTooltip
                                content="Purple boxes are the longest lanes detected. Follows the invisible snake theory."
                                placement="bottom"
                            >
                                <i class="fas fa-info-circle" style={{ float: "right" }} />
                            </CTooltip>
                        </CCardHeader>
                        <CCardBody>
                            <div>
                                <img
                                    src={predictImage}
                                    alt="Predicted Image"
                                    style={{ width: "100%", height: "auto" }}
                                />
                            </div>
                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={5}>
                    <CCard className="mb-12">
                        <CCardBody>
                            Related Roads
                            <QueueTable queueData={currQueue} />
                            <DensityMeter level={currDensity} />

                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={6}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Queue Length</strong>
                        </CCardHeader>
                        <CCardBody>
                            {/* {queueData} */}
                            {/* <QueueTable queueData={queueData} /> */}
                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={6}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Traffic Density</strong>
                        </CCardHeader>
                        <CCardBody>
                            {/* <DensityMeter level={level} /> */}
                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={12}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Traffic Profiles</strong>
                            <CTooltip
                                content="Information is updated on a weekly basis"
                                placement="bottom"
                            >
                                <i class="fas fa-info-circle" style={{ float: "right" }} />
                            </CTooltip>
                        </CCardHeader>
                        <CCardBody>
                        </CCardBody>
                    </CCard>
                </CCol>
            </CRow>

        </>
    )
}

export default Metrics
