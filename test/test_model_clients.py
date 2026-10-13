import numpy as np

from fluxthemis.model_clients.fluxvla_policy import FluxVLAPolicyClient
from fluxthemis.reporters.fluxvla_reporter import FluxVLAEvaluationReporter
from fluxthemis.structures import TaskSpec
from fluxthemis.transports.zmq_protocol import (decode_frame, decode_header,
                                                encode_frame, encode_header)


class _FakeTransport:

    def __init__(self, response_type):
        self.response_type = response_type
        self.server_deployment_id = 'server-deployment'
        self.request_timeout_s = 5.0
        self.setup_requirements = []
        self.requests = []
        self.closed = False

    def setup(self, *, require_evaluation_reporting=False):
        self.setup_requirements.append(require_evaluation_reporting)

    def request(self, header, payload):
        request_header = decode_header(header)
        request_payload = decode_frame(payload)
        self.requests.append((request_header, request_payload))
        request_id = request_header['request_id']
        if self.response_type == 'predict_response':
            return (
                decode_header(
                    encode_header(
                        'predict_response',
                        request_id=request_id,
                        ok=True,
                        inference_time_s=0.25,
                    )),
                encode_frame({'actions': np.ones((2, 3), dtype=np.float32)}),
            )
        return (
            decode_header(
                encode_header(
                    'report_response', request_id=request_id, ok=True)),
            encode_frame({
                'accepted': True,
                'run_dir': '/tmp/evaluation',
            }),
        )

    @staticmethod
    def validate_response(header, request_id):
        assert header['request_id'] == request_id
        assert header['ok'] is True

    def close(self):
        self.closed = True


def test_policy_client_only_handles_episode_inference():
    client = FluxVLAPolicyClient(transport={
        'image_keys': ['image'],
        'state_keys': ['state'],
    })
    transport = _FakeTransport('predict_response')
    client.transport = transport

    client.setup()
    client.reset(TaskSpec('pick'), episode_index=2, policy_seed=11)
    client.set_episode_id('session:3')
    chunk = client.predict_action(
        {
            'image': np.zeros((4, 4, 3), dtype=np.uint8),
            'state': np.zeros(7),
        }, {
            'task_description': 'pick the object',
        })

    header, payload = transport.requests[0]
    assert transport.setup_requirements == [False]
    assert header['type'] == 'predict_action'
    assert header['policy_seed'] == 11
    assert payload['task_description'] == 'pick the object'
    assert chunk.actions.shape == (2, 3)
    assert chunk.metadata['episode_id'] == 'session:3'
    assert not hasattr(client, 'report_evaluation')


def test_reporter_only_handles_evaluation_lifecycle():
    reporter = FluxVLAEvaluationReporter(transport={})
    transport = _FakeTransport('report_response')
    reporter.transport = transport

    reporter.setup()
    run_session_id = reporter.run_session_id
    run_dir = reporter.report_evaluation('run_start', {'base_seed': 7})
    reporter.report_evaluation('run_end', {'status': 'finished'})

    first_header, _ = transport.requests[0]
    second_header, _ = transport.requests[1]
    assert transport.setup_requirements == [True]
    assert first_header['run_session_id'] == run_session_id
    assert first_header['sequence'] == 1
    assert second_header['sequence'] == 2
    assert run_dir == '/tmp/evaluation'
    assert reporter.evaluation_report_run_dir == '/tmp/evaluation'
    assert not hasattr(reporter, 'predict_action')
