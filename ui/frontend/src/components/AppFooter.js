import React from 'react'
import { CFooter } from '@coreui/react'

const AppFooter = () => {
  return (
    <CFooter className="px-4">
      <div>
        <a href="https://www.smu.edu.sg/" target="_blank" rel="noopener noreferrer">
          AI-Driven Image Analysis for Urban Traffic Insights
        </a>
        <span className="ms-1">&copy; SMU CS480 CAM-R</span>
      </div>
      <div className="ms-auto">
        <span className="me-1">Disclaimer: Metrics are algorithmically derived by AI and may not be accurate</span>
        {/* <a href="https://coreui.io/react" target="_blank" rel="noopener noreferrer">
          CoreUI React Admin &amp; Dashboard Template
        </a> */}
      </div>
    </CFooter>
  )
}

export default React.memo(AppFooter)
