(() => {
  const fullTurn = 2 * Math.PI;

  class FullOrbitControls {
    constructor(camera, element) {
      this.camera = camera;
      this.element = element;
      this.target = new THREE.Vector3();
      this.offset = new THREE.Vector3();
      this.right = new THREE.Vector3();
      this.up = new THREE.Vector3();
      this.minDistance = 0;
      this.maxDistance = Infinity;
      this.enableDamping = true;
      this.dampingFactor = 0.15;
      this.azimuth = 0;
      this.elevation = 0;
      this.distance = 1;
      this.pointers = new Map();

      element.style.touchAction = 'none';
      element.addEventListener('pointerdown', event => this.startPointer(event));
      element.addEventListener('pointermove', event => this.movePointer(event));
      for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        element.addEventListener(type, event => this.pointers.delete(event.pointerId));
      }
      element.addEventListener('contextmenu', event => event.preventDefault());
      element.addEventListener('wheel', event => this.wheel(event), { passive: false });
      this.sync();
    }

    // Resizing and fitting keep the current orbit. A view preset starts upright.
    sync(preserveOrbit = false) {
      this.offset.copy(this.camera.position).sub(this.target);
      this.distance = this.offset.length() || this.minDistance || 1;
      const horizontal = Math.hypot(this.offset.x, this.offset.z);
      const inverted = preserveOrbit && Math.cos(this.elevation) < 0;
      const elevation = Math.atan2(this.offset.y, horizontal);
      this.elevation = inverted ? Math.PI - elevation : elevation;
      if (horizontal > 1e-8) {
        this.azimuth = Math.atan2(this.offset.x, this.offset.z) + (inverted ? Math.PI : 0);
      } else if (!preserveOrbit) {
        this.azimuth = 0;
      }
      this.yawDelta = 0;
      this.pitchDelta = 0;
      this.panX = 0;
      this.panY = 0;
      this.zoomDelta = 0;
      this.update();
    }

    update() {
      const fraction = this.enableDamping ? this.dampingFactor : 1;
      this.azimuth = (this.azimuth + this.yawDelta * fraction) % fullTurn;
      this.elevation = (this.elevation + this.pitchDelta * fraction) % fullTurn;
      this.distance = Math.max(this.minDistance,
        Math.min(this.maxDistance, this.distance * Math.exp(this.zoomDelta * fraction)));

      const sinAzimuth = Math.sin(this.azimuth);
      const cosAzimuth = Math.cos(this.azimuth);
      const sinElevation = Math.sin(this.elevation);
      const cosElevation = Math.cos(this.elevation);
      // This up vector is tangent to the orbit, including at either pole.
      this.up.set(-sinElevation * sinAzimuth, cosElevation, -sinElevation * cosAzimuth);
      this.right.set(cosAzimuth, 0, -sinAzimuth);

      if (this.panX || this.panY) {
        const height = this.element.clientHeight || 1;
        const scale = 2 * this.distance * Math.tan(this.camera.fov * Math.PI / 360)
          / (height * this.camera.zoom);
        this.target.addScaledVector(this.right, -this.panX * fraction * scale);
        this.target.addScaledVector(this.up, this.panY * fraction * scale);
      }

      this.offset.set(cosElevation * sinAzimuth, sinElevation, cosElevation * cosAzimuth);
      this.camera.position.copy(this.target).addScaledVector(this.offset, this.distance);
      this.camera.up.copy(this.up);
      this.camera.lookAt(this.target);

      const remaining = 1 - fraction;
      this.yawDelta *= remaining;
      this.pitchDelta *= remaining;
      this.panX *= remaining;
      this.panY *= remaining;
      this.zoomDelta *= remaining;
    }

    startPointer(event) {
      if (event.pointerType !== 'touch' && ![0, 1, 2].includes(event.button)) return;
      this.pointers.set(event.pointerId, {
        type: event.pointerType,
        button: event.button,
        x: event.clientX,
        y: event.clientY
      });
      this.element.setPointerCapture(event.pointerId);
    }

    movePointer(event) {
      const pointer = this.pointers.get(event.pointerId);
      if (!pointer) return;
      const dx = event.clientX - pointer.x;
      const dy = event.clientY - pointer.y;
      const touches = [...this.pointers.values()].filter(pointer => pointer.type === 'touch');
      if (pointer.type === 'touch' && touches.length > 1) {
        const [first, second] = touches;
        if (pointer === first || pointer === second) {
          const other = pointer === first ? second : first;
          const before = Math.hypot(pointer.x - other.x, pointer.y - other.y);
          const after = Math.hypot(event.clientX - other.x, event.clientY - other.y);
          if (before > 0 && after > 0) this.zoomDelta += Math.log(before / after);
          this.panX += dx / 2;
          this.panY += dy / 2;
        }
      } else if (pointer.type !== 'touch' && pointer.button === 1) {
        this.zoomDelta += dy * 0.01;
      } else if (pointer.type !== 'touch' &&
                 (pointer.button === 2 || event.ctrlKey || event.metaKey || event.shiftKey)) {
        this.panX += dx;
        this.panY += dy;
      } else {
        const radiansPerPixel = fullTurn / (this.element.clientHeight || 1);
        const inverted = Math.cos(this.elevation) < 0;
        this.yawDelta += dx * radiansPerPixel * (inverted ? 1 : -1);
        this.pitchDelta += dy * radiansPerPixel;
      }
      pointer.x = event.clientX;
      pointer.y = event.clientY;
    }

    wheel(event) {
      if (event.deltaY === 0) return;
      event.preventDefault();
      const multiplier = event.deltaMode === 1 ? 16
        : event.deltaMode === 2 ? this.element.clientHeight
        : 1;
      this.zoomDelta += event.deltaY * multiplier * 0.0012;
    }
  }

  THREE.FullOrbitControls = FullOrbitControls;
})();
