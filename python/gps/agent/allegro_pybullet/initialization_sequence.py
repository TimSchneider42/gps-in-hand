from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body.allegro_hand import AllegroRightHand


class InitializationSequence:
    def on_initialize(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        pass

    def run(self, physics_client: PhysicsClient, hand: AllegroRightHand):
        pass
