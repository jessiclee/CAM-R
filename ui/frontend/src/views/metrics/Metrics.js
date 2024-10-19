import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
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

    // Timestamp for Image
    const [timestamp, setTimestamp] = useState('');
    useEffect(() => {
        const localTimestamp = new Date().toLocaleString();
        setTimestamp(localTimestamp);
    }, []);
    // Image
    // Queue Length
    // Density
    // Meter

    // MOCK THE API RETRIEVAL FIRST
    const densityJson = {2703:"Low"}
    const currDensity = densityJson[id]
    // const currDensity = "Low"  //hard code first
    const [level, setLevel] = useState(currDensity);

    return (
        <>
            <h1>Metrics: {id}</h1>
            <CRow>
                <CCol xs={8}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Current View @ {timestamp}</strong>
                            <CTooltip
                                content="Purple boxes are the longest lanes detected. Follows the invisible snake theory."
                                placement="bottom"
                            >
                                <i class="fas fa-info-circle" style={{ float: "right" }}/>
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
                <CCol xs={4}>
                    <CCard className="mb-12">
                        <CCardBody>
                            Related Roads
                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={6}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Queue Length</strong>
                        </CCardHeader>
                        <CCardBody>
                            <QueueTable />
                        </CCardBody>
                    </CCard>
                </CCol>
                <CCol xs={6}>
                    <CCard className="mb-4">
                        <CCardHeader>
                            <strong>Traffic Density</strong>
                        </CCardHeader>
                        <CCardBody>
                            <DensityMeter level={level} />
                        </CCardBody>
                    </CCard>
                </CCol>
            </CRow>

        </>
    )
}

export default Metrics
