import { useRef, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Float, Environment } from '@react-three/drei';
import * as THREE from 'three';

// Global mouse position store
const mousePosition = { x: 0, y: 0 };

function BinocularEye({ position, side }: { position: [number, number, number]; side: 'left' | 'right' }) {
  const eyeGroupRef = useRef<THREE.Group>(null);
  const targetRef = useRef(new THREE.Vector3());
  const currentLookAt = useRef(new THREE.Vector3(0, 0, 5));

  useFrame(({ camera }) => {
    if (eyeGroupRef.current) {
      // Calculate 3D target position from mouse coordinates
      // Project the mouse position onto a plane in front of the robot
      const targetX = mousePosition.x * 3; // Spread across wider area
      const targetY = mousePosition.y * 2 + 0.5; // Offset for robot height
      const targetZ = 5; // Fixed distance in front

      targetRef.current.set(targetX, targetY, targetZ);

      // Smooth interpolation to target
      currentLookAt.current.lerp(targetRef.current, 0.12);

      // Get world position of the eye
      const eyeWorldPos = new THREE.Vector3();
      eyeGroupRef.current.getWorldPosition(eyeWorldPos);

      // Calculate direction to look at
      const direction = new THREE.Vector3()
        .subVectors(currentLookAt.current, eyeWorldPos)
        .normalize();

      // Convert direction to rotation
      const targetRotationY = Math.atan2(direction.x, direction.z);
      const targetRotationX = -Math.asin(direction.y);

      // Apply with lerp for smoothness
      eyeGroupRef.current.rotation.y = THREE.MathUtils.lerp(
        eyeGroupRef.current.rotation.y,
        targetRotationY * 0.6,
        0.15
      );
      eyeGroupRef.current.rotation.x = THREE.MathUtils.lerp(
        eyeGroupRef.current.rotation.x,
        targetRotationX * 0.6,
        0.15
      );
    }
  });

  return (
    <group position={position}>
      {/* Eye stalk/neck */}
      <mesh position={[0, -0.08, 0]}>
        <cylinderGeometry args={[0.03, 0.04, 0.12, 16]} />
        <meshStandardMaterial color="#4a4a4a" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Binocular housing */}
      <group ref={eyeGroupRef}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.09, 0.11, 0.14, 32]} />
          <meshStandardMaterial color="#2a2a2a" metalness={0.9} roughness={0.2} />
        </mesh>

        {/* Lens rim */}
        <mesh position={[0, 0, 0.07]}>
          <torusGeometry args={[0.08, 0.015, 16, 32]} />
          <meshStandardMaterial color="#1a1a1a" metalness={0.9} roughness={0.1} />
        </mesh>

        {/* Lens glass - glowing */}
        <mesh position={[0, 0, 0.07]}>
          <circleGeometry args={[0.075, 32]} />
          <meshStandardMaterial
            color="#87CEEB"
            emissive="#87CEEB"
            emissiveIntensity={0.6}
          />
        </mesh>

        {/* Pupil/iris */}
        <mesh position={[0, 0, 0.075]}>
          <circleGeometry args={[0.04, 32]} />
          <meshStandardMaterial color="#1a1a1a" />
        </mesh>

        {/* Eye highlight */}
        <mesh position={[0.02, 0.02, 0.08]}>
          <circleGeometry args={[0.012, 16]} />
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.8} />
        </mesh>

        {/* Secondary highlight */}
        <mesh position={[-0.015, -0.02, 0.08]}>
          <circleGeometry args={[0.006, 16]} />
          <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.5} />
        </mesh>
      </group>
    </group>
  );
}

function WallEHead() {
  const headRef = useRef<THREE.Group>(null);
  const neckRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (headRef.current) {
      // Responsive head following
      const targetY = mousePosition.x * 0.35;
      const targetX = -mousePosition.y * 0.25;

      headRef.current.rotation.y = THREE.MathUtils.lerp(
        headRef.current.rotation.y,
        targetY,
        0.08 // Faster for responsive movement
      );
      headRef.current.rotation.x = THREE.MathUtils.lerp(
        headRef.current.rotation.x,
        targetX,
        0.08
      );

      // Gentle floating
      headRef.current.position.y = 0.55 + Math.sin(state.clock.elapsedTime * 0.5) * 0.01;
    }

    if (neckRef.current) {
      // Neck extends slightly based on looking direction
      neckRef.current.scale.y = 1 + Math.abs(mousePosition.y) * 0.1;
    }
  });

  return (
    <group ref={headRef} position={[0, 0.55, 0.1]}>
      {/* Neck - accordion/flexible style */}
      <mesh ref={neckRef} position={[0, -0.12, -0.05]}>
        <cylinderGeometry args={[0.06, 0.08, 0.15, 16]} />
        <meshStandardMaterial color="#3a3a3a" metalness={0.7} roughness={0.4} />
      </mesh>

      {/* Eye bar/mount connecting both eyes */}
      <mesh position={[0, 0.02, 0]}>
        <boxGeometry args={[0.22, 0.05, 0.06]} />
        <meshStandardMaterial color="#3a3a3a" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Left Eye */}
      <BinocularEye position={[-0.1, 0.02, 0.03]} side="left" />

      {/* Right Eye */}
      <BinocularEye position={[0.1, 0.02, 0.03]} side="right" />
    </group>
  );
}

function WallEBody() {
  const bodyRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (bodyRef.current) {
      // Very subtle body sway
      bodyRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.02;
    }
  });

  return (
    <group ref={bodyRef}>
      {/* Main body - cube/compactor shape */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.6, 0.5, 0.5]} />
        <meshStandardMaterial color="#D4A017" metalness={0.4} roughness={0.6} />
      </mesh>

      {/* Body front panel */}
      <mesh position={[0, 0, 0.251]}>
        <boxGeometry args={[0.55, 0.45, 0.01]} />
        <meshStandardMaterial color="#C4920F" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Compactor door lines */}
      {[-0.1, 0.1].map((y, i) => (
        <mesh key={i} position={[0, y, 0.26]}>
          <boxGeometry args={[0.5, 0.02, 0.005]} />
          <meshStandardMaterial color="#8B7500" metalness={0.6} roughness={0.4} />
        </mesh>
      ))}

      {/* Side panels with texture */}
      {[-0.301, 0.301].map((x, i) => (
        <mesh key={i} position={[x, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
          <boxGeometry args={[0.45, 0.45, 0.01]} />
          <meshStandardMaterial color="#B8930E" metalness={0.5} roughness={0.6} />
        </mesh>
      ))}

      {/* Top lid */}
      <mesh position={[0, 0.26, 0]}>
        <boxGeometry args={[0.62, 0.03, 0.52]} />
        <meshStandardMaterial color="#8B7500" metalness={0.5} roughness={0.5} />
      </mesh>

      {/* Solar panel collectors on top */}
      <mesh position={[0, 0.29, -0.1]}>
        <boxGeometry args={[0.4, 0.02, 0.25]} />
        <meshStandardMaterial color="#2a2a2a" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Small details - rivets */}
      {[
        [-0.25, 0.18, 0.252],
        [0.25, 0.18, 0.252],
        [-0.25, -0.18, 0.252],
        [0.25, -0.18, 0.252],
      ].map((pos, i) => (
        <mesh key={i} position={pos as [number, number, number]}>
          <cylinderGeometry args={[0.015, 0.015, 0.02, 8]} />
          <meshStandardMaterial color="#5a5a5a" metalness={0.9} roughness={0.2} />
        </mesh>
      ))}
    </group>
  );
}

function WallEArms() {
  const leftArmRef = useRef<THREE.Group>(null);
  const rightArmRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (leftArmRef.current) {
      leftArmRef.current.rotation.z = -0.3 + Math.sin(state.clock.elapsedTime * 0.6) * 0.08;
      leftArmRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.4) * 0.05;
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.z = 0.3 + Math.sin(state.clock.elapsedTime * 0.6 + 1) * 0.08;
      rightArmRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.4 + 1) * 0.05;
    }
  });

  const ArmSegment = ({ isLeft }: { isLeft: boolean }) => (
    <group>
      {/* Shoulder mount */}
      <mesh>
        <boxGeometry args={[0.06, 0.06, 0.06]} />
        <meshStandardMaterial color="#4a4a4a" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Upper arm */}
      <mesh position={[isLeft ? -0.08 : 0.08, -0.06, 0]}>
        <boxGeometry args={[0.04, 0.14, 0.04]} />
        <meshStandardMaterial color="#5a5a5a" metalness={0.7} roughness={0.4} />
      </mesh>

      {/* Elbow joint */}
      <mesh position={[isLeft ? -0.08 : 0.08, -0.15, 0]}>
        <sphereGeometry args={[0.03, 16, 16]} />
        <meshStandardMaterial color="#3a3a3a" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Lower arm */}
      <mesh position={[isLeft ? -0.08 : 0.08, -0.25, 0]}>
        <boxGeometry args={[0.035, 0.14, 0.035]} />
        <meshStandardMaterial color="#5a5a5a" metalness={0.7} roughness={0.4} />
      </mesh>

      {/* Gripper/claw */}
      <group position={[isLeft ? -0.08 : 0.08, -0.35, 0]}>
        <mesh>
          <boxGeometry args={[0.05, 0.03, 0.04]} />
          <meshStandardMaterial color="#3a3a3a" metalness={0.8} roughness={0.3} />
        </mesh>
        {/* Claw fingers */}
        <mesh position={[-0.02, -0.03, 0]} rotation={[0, 0, 0.2]}>
          <boxGeometry args={[0.015, 0.05, 0.02]} />
          <meshStandardMaterial color="#4a4a4a" metalness={0.8} roughness={0.3} />
        </mesh>
        <mesh position={[0.02, -0.03, 0]} rotation={[0, 0, -0.2]}>
          <boxGeometry args={[0.015, 0.05, 0.02]} />
          <meshStandardMaterial color="#4a4a4a" metalness={0.8} roughness={0.3} />
        </mesh>
      </group>
    </group>
  );

  return (
    <>
      <group ref={leftArmRef} position={[-0.33, 0.1, 0.15]}>
        <ArmSegment isLeft={true} />
      </group>
      <group ref={rightArmRef} position={[0.33, 0.1, 0.15]}>
        <ArmSegment isLeft={false} />
      </group>
    </>
  );
}

function WallETreads() {
  const leftTreadRef = useRef<THREE.Group>(null);
  const rightTreadRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    // Subtle tread movement
    if (leftTreadRef.current && rightTreadRef.current) {
      const wobble = Math.sin(state.clock.elapsedTime * 0.3) * 0.01;
      leftTreadRef.current.position.y = -0.35 + wobble;
      rightTreadRef.current.position.y = -0.35 - wobble;
    }
  });

  const Tread = () => (
    <group>
      {/* Tread housing */}
      <mesh>
        <boxGeometry args={[0.12, 0.18, 0.45]} />
        <meshStandardMaterial color="#2a2a2a" metalness={0.7} roughness={0.5} />
      </mesh>

      {/* Track texture - ridges */}
      {[-0.18, -0.09, 0, 0.09, 0.18].map((z, i) => (
        <mesh key={i} position={[0.061, 0, z]}>
          <boxGeometry args={[0.01, 0.16, 0.04]} />
          <meshStandardMaterial color="#1a1a1a" metalness={0.8} roughness={0.4} />
        </mesh>
      ))}

      {/* Wheels inside tread */}
      {[-0.15, 0.15].map((z, i) => (
        <mesh key={i} position={[0, 0, z]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.07, 0.07, 0.1, 16]} />
          <meshStandardMaterial color="#3a3a3a" metalness={0.8} roughness={0.3} />
        </mesh>
      ))}
    </group>
  );

  return (
    <>
      <group ref={leftTreadRef} position={[-0.28, -0.35, 0]}>
        <Tread />
      </group>
      <group ref={rightTreadRef} position={[0.28, -0.35, 0]}>
        <Tread />
      </group>
    </>
  );
}

function GlowEffects() {
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (ringRef.current) {
      ringRef.current.rotation.z = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <>
      {/* Floor shadow/glow */}
      <mesh position={[0, -0.55, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.8, 32]} />
        <meshStandardMaterial
          color="#87CEEB"
          emissive="#87CEEB"
          emissiveIntensity={0.15}
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* Ambient ring */}
      <mesh ref={ringRef} position={[0, -0.3, 0]}>
        <torusGeometry args={[1.0, 0.008, 16, 64]} />
        <meshStandardMaterial
          color="#87CEEB"
          emissive="#87CEEB"
          emissiveIntensity={0.4}
          transparent
          opacity={0.4}
        />
      </mesh>
    </>
  );
}

function MouseTracker() {
  const { size } = useThree();

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      mousePosition.x = (event.clientX / size.width) * 2 - 1;
      mousePosition.y = -((event.clientY / size.height) * 2 - 1);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [size]);

  return null;
}

function Scene() {
  return (
    <>
      <MouseTracker />
      <ambientLight intensity={0.5} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#FFF5E1" />
      <pointLight position={[-5, 3, 5]} intensity={0.4} color="#87CEEB" />
      <spotLight
        position={[0, 5, 3]}
        angle={0.5}
        penumbra={1}
        intensity={1.2}
        color="#ffffff"
      />

      <Float speed={0.8} rotationIntensity={0.05} floatIntensity={0.15}>
        <group position={[0, 0.1, 0]} scale={0.95}>
          <WallEBody />
          <WallEHead />
          <WallEArms />
          <WallETreads />
        </group>
      </Float>

      <GlowEffects />

      <Environment preset="sunset" />
    </>
  );
}

export default function Robot3D() {
  return (
    <div className="w-full h-[350px] md:h-[400px]">
      <Canvas
        camera={{ position: [0, 0.3, 2.0], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Scene />
      </Canvas>
    </div>
  );
}