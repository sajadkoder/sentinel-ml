"""
Dashboard API endpoints for monitoring
"""
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException

from sentinel_ml.core.logging import get_logger
from sentinel_ml.monitoring import MonitoringDashboard

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])

dashboard = MonitoringDashboard()


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get current monitoring metrics"""
    return dashboard.collect_metrics()


@router.get("/drift")
async def get_drift_summary() -> Dict[str, Any]:
    """Get data drift detection summary"""
    return dashboard.drift_detector.get_drift_summary()


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get model performance metrics"""
    return dashboard.performance_monitor.compute_metrics()


@router.get("/alerts")
async def get_alerts(hours: int = 24) -> List[Dict]:
    """Get recent alerts"""
    return dashboard.alerting.get_recent_alerts(hours=hours)


@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """Get overall system health status"""
    metrics = dashboard.collect_metrics()
    return {
        "status": metrics.get("system_health", "unknown"),
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "model": "healthy" if metrics.get("performance") else "unknown",
            "drift_detection": "healthy",
            "alerting": "healthy"
        }
    }
