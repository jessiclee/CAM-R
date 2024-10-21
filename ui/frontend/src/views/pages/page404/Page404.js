import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CCol,
  CContainer,
  CRow,
} from '@coreui/react'

const Page404 = () => {

  const navigate = useNavigate();

  // Redirect to main page after 3 seconds
  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/');
    }, 5000);

    // Cleanup the timer when component unmounts
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div>
    {/* <div className="bg-body-tertiary min-vh-100 d-flex flex-row align-items-center"> */}
      <CContainer>
        <CRow className="justify-content-center">
          <CCol md={6}>
            <div className="clearfix">
              <h1 className="float-start display-3 me-4">404</h1>
              <h4 className="pt-3">Oops! You{"'"}re lost.</h4>
              <p className="text-body-secondary float-start">
                The page you are looking for was not found. Redirecting back to Dashboard...
              </p>
            </div>
          </CCol>
        </CRow>
      </CContainer>
    </div>
  )
}

export default Page404
