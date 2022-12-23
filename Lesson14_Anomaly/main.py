import AnomalyService

print("Elliptic Envelope")
AnomalyService.EnvelopeAnomalys()
print("DBSCAN")
AnomalyService.DBSCANAnomalys()
print("Isolation Forest")
AnomalyService.IsolationForestAnomalys()
print("One SVM")
AnomalyService.OneSVMAnomalys()