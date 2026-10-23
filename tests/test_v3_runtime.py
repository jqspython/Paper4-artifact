"""Changed Kubernetes adapter paths with a fake API; no containers or traffic."""
import unittest
from unittest.mock import patch

from sysbench.control import ContractMonitor
from sysbench.runtime import PodRuntime, cost_between


class API:
    def __init__(self): self.items={}; self.hidden=False
    def call(self,method,resource,body=None):
        if method=="GET": return {"items":[] if self.hidden else list(self.items.values())}
        if method=="POST":
            name=body["metadata"]["name"]
            result={"metadata":{"name":name,"uid":name},"status":{"conditions":[]}}
            self.items[name]=result
            return result
        if method=="DELETE": self.items.pop(resource.split("/")[-1]); return {}
        raise AssertionError(method)
    def ready(self,name,value=True):
        self.items[name]["status"]["conditions"]=[{"type":"Ready","status":"True" if value else "False"}]


class RuntimeProperties(unittest.TestCase):
    def test_uncertain_post_is_cleaned_by_owned_name_without_claiming_valid_billing(self):
        class UncertainAPI(API):
            def call(self,method,resource,body=None):
                result=super().call(method,resource,body)
                if method=="POST": raise TimeoutError("created server side, response lost")
                return result
        api=UncertainAPI()
        runtime=PodRuntime(api,"uncertain",ContractMonitor(96))
        with self.assertRaises(TimeoutError): runtime.create(5,0)
        self.assertEqual(len(api.items),1)
        runtime.close()
        self.assertFalse(api.items)
        self.assertFalse(runtime.uncertain)
        self.assertFalse(runtime.events[0]["accepted"])
        self.assertEqual(runtime.events[-1]["type"],"uncertain_create_cleanup")

    def test_accepted_not_visible_is_inflight_but_seen_missing_or_ready_regression_is_invalid(self):
        api=API(); clock=[10.]
        with patch("sysbench.runtime.time.monotonic",side_effect=lambda:clock[0]):
            runtime=PodRuntime(api,"synthetic",ContractMonitor(96,clock=lambda:clock[0]))
            name=runtime.create(5,0); api.hidden=True
            self.assertEqual(runtime.state(),(0,1))
            with self.assertRaises(ValueError): runtime.delete(name)
            api.hidden=False; clock[0]=12; self.assertEqual(runtime.state(),(0,1))
            api.hidden=True
            with self.assertRaises(RuntimeError): runtime.state()
            api.hidden=False; api.ready(name); clock[0]=20
            self.assertEqual(runtime.state(),(1,0))
            api.ready(name,False)
            with self.assertRaises(RuntimeError): runtime.state()

    def test_paid_pending_ready_costs_lag_and_bounded_issue(self):
        api=API(); clock=[10.]
        with patch("sysbench.runtime.time.monotonic",side_effect=lambda:clock[0]):
            runtime=PodRuntime(api,"synthetic",ContractMonitor(96,clock=lambda:clock[0]))
            name=runtime.create(5,0)
            with self.assertRaises(ValueError): runtime.issue(4,5,0)
            api.ready(name); clock[0]=20
            self.assertEqual(runtime.service_start(8),1)
            self.assertEqual(runtime.events[-1]["discrete_lag"],8)
            clock[0]=30; runtime.close()
            self.assertEqual(cost_between(runtime.events,0,40),20)
            self.assertEqual(cost_between(runtime.events,0,40,ready=True),10)
            self.assertEqual(cost_between(runtime.events,15,25),10)


if __name__=="__main__": unittest.main()
