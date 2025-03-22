#![feature(never_type)]
#![feature(associated_type_defaults)]
#![feature(coroutines, coroutine_trait, stmt_expr_attributes, iter_from_coroutine)]

pub trait GeometricSpace {
	type Point;
	type Vector;
	type Line;
	type Ray;
	type Plane;

	fn radius_vector(p: Self::Point) -> Self::Vector;
	fn line_between(a: Self::Point, b: Self::Point) -> Self::Line;
}

pub trait ApproxEq: Copy + PartialEq {
	fn approx_eq(self, other: Self) -> bool;
}

pub type R = f32;

impl ApproxEq for R {
	fn approx_eq(self, other: Self) -> bool {
		(self - other).abs() <= 1e-7
	}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pnt3 {
	pub x: R,
	pub y: R,
	pub z: R,
}

impl Pnt3 {
	pub const fn new(x: R, y: R, z: R) -> Self {
		Self { x, y, z }
	}

	pub const fn zero() -> Self {
		Self::new(0.0, 0.0, 0.0)
	}

	pub const fn dot(self, o: Self) -> R {
		self.x * o.x + self.y * o.y + self.z * o.z
	}

	/// Project self onto a unit sphere (normalize)
	pub fn onto_sphere(self, radius: R) -> Self {
		if self.approx_eq(Pnt3::zero()) {
			Pnt3::zero()
		} else {
			let k = radius / (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
			Self {
				x: self.x * k,
				y: self.y * k,
				z: self.z * k,
			}
		}
	}
}

impl ApproxEq for Pnt3 {
	fn approx_eq(self, other: Self) -> bool {
		self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
	}
}

impl std::ops::Add<Vct3> for Pnt3 {
	type Output = Pnt3;

	fn add(self, rhs: Vct3) -> Self::Output {
		Pnt3 {
			x: self.x + rhs.x,
			y: self.y + rhs.y,
			z: self.z + rhs.z,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vct3 {
	pub x: R,
	pub y: R,
	pub z: R,
}

impl std::ops::Mul<R> for Vct3 {
	type Output = Self;

	fn mul(self, rhs: R) -> Self::Output {
		Self {
			x: self.x * rhs,
			y: self.y * rhs,
			z: self.z * rhs,
		}
	}
}

impl Vct3 {
	pub fn dot(self, o: Self) -> R {
		self.x * o.x + self.y * o.y + self.z * o.z
	}

	pub fn cross(self, b: Self) -> Self {
		let a = self;
		Self {
			x: a.y * b.z - a.z * b.y,
			y: a.z * b.x - a.x * b.z,
			z: a.x * b.y - a.y * b.x,
		}
	}

	pub const fn new(x: R, y: R, z: R) -> Self {
		Self { x, y, z }
	}

	pub const fn zero() -> Self {
		Self::new(0.0, 0.0, 0.0)
	}

	pub const fn nth(&self, n: usize) -> &R {
		match n {
			0 => &self.x,
			1 => &self.y,
			2 => &self.z,
			_ => unreachable!(),
		}
	}

	pub const fn nth_mut(&mut self, n: usize) -> &mut R {
		match n {
			0 => &mut self.x,
			1 => &mut self.y,
			2 => &mut self.z,
			_ => unreachable!(),
		}
	}
}

impl ApproxEq for Vct3 {
	fn approx_eq(self, other: Self) -> bool {
		self.x.approx_eq(other.x) && self.y.approx_eq(other.y) && self.z.approx_eq(other.z)
	}
}

impl std::ops::Sub<Pnt3> for Pnt3 {
	type Output = Vct3;

	fn sub(self, rhs: Pnt3) -> Self::Output {
		Vct3 {
			x: self.x - rhs.x,
			y: self.y - rhs.y,
			z: self.z - rhs.z,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pnt2 {
	pub x: R,
	pub y: R,
}

impl Pnt2 {
	pub fn is_inside(self, min: Pnt2, max: Pnt2) -> bool {
		min.x <= self.x && self.x <= max.x && min.y <= self.y && self.y <= max.y
	}
}

impl std::ops::Sub<Pnt2> for Pnt2 {
	type Output = Vct2;

	fn sub(self, rhs: Pnt2) -> Self::Output {
		Vct2 {
			x: self.x - rhs.x,
			y: self.y - rhs.y,
		}
	}
}

impl std::ops::Add<Vct2> for Pnt2 {
	type Output = Pnt2;

	fn add(self, rhs: Vct2) -> Self::Output {
		Pnt2 {
			x: self.x + rhs.x,
			y: self.y + rhs.y,
		}
	}
}

impl ApproxEq for Pnt2 {
	fn approx_eq(self, other: Self) -> bool {
		self.x.approx_eq(other.x) && self.y.approx_eq(other.y)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vct2 {
	pub x: R,
	pub y: R,
}

impl Vct2 {
	/// Computes the dot product of two vectors.
	/// Note that this operation is commutative: `a.dot(b) == b.dot(a)`.
	pub const fn dot(self, o: Self) -> R {
		self.x * o.x + self.y * o.y
	}

	/// Magnitude squared. Same as `self.dot(self)`.
	pub const fn magnitude2(self) -> R {
		self.dot(self)
	}

	/// Magnitude of the vector, same as `self.magnitude2().sqrt()`.
	pub fn magnitude(self) -> R {
		self.dot(self).sqrt()
	}
}

impl std::ops::Mul<R> for Vct2 {
	type Output = Self;

	fn mul(self, rhs: R) -> Self::Output {
		Self {
			x: self.x * rhs,
			y: self.y * rhs,
		}
	}
}

impl ApproxEq for Vct2 {
	fn approx_eq(self, other: Self) -> bool {
		self.x.approx_eq(other.x) && self.y.approx_eq(other.y)
	}
}

pub struct GSpace3D;

impl GeometricSpace for GSpace3D {
	type Point = Pnt3;
	type Vector = Vct3;
	type Line = (Pnt3, Vct3); // TODO: better defn.
	type Ray = (Pnt3, Vct3);
	type Plane = (Vct3, R);

	fn radius_vector(p: Self::Point) -> Self::Vector {
		Vct3 {
			x: p.x,
			y: p.y,
			z: p.z,
		}
	}

	fn line_between(a: Self::Point, b: Self::Point) -> Self::Line {
		(a, b - a)
	}
}

pub struct GSpace2D;

impl GeometricSpace for GSpace2D {
	type Point = Pnt2;
	type Vector = Vct2;
	// type Line = (Vct2, R);
	type Line = (Pnt2, Vct2);
	type Ray = (Pnt2, Vct2);
	type Plane = ();

	fn radius_vector(p: Self::Point) -> Self::Vector {
		Vct2 { x: p.x, y: p.y }
	}

	fn line_between(a: Self::Point, b: Self::Point) -> Self::Line {
		(a, b - a)
	}
}

pub struct GSpace1D;

impl GeometricSpace for GSpace1D {
	type Point = R;
	type Vector = R;
	type Line = ();
	type Ray = (R, bool);
	type Plane = !;

	fn radius_vector(p: Self::Point) -> Self::Vector {
		p
	}

	fn line_between(_: Self::Point, _: Self::Point) -> Self::Line {
		()
	}
}

pub trait GeometricMap<In: GeometricSpace> {
	type Out: GeometricSpace;

	fn project_point(&self, p: In::Point) -> <Self::Out as GeometricSpace>::Point;

	fn project_vector(&self, p: In::Vector) -> <Self::Out as GeometricSpace>::Vector;

	fn project_line(&self, p: In::Line) -> <Self::Out as GeometricSpace>::Line;

	fn project_ray(&self, p: In::Ray) -> <Self::Out as GeometricSpace>::Ray;

	/// Get direction from point to camera.
	fn get_back_dir(&self, p: In::Point) -> In::Vector;
}

/// Marker trait that specifies that a map is translation-invariant-ish.
/// That means that `|C - Project(C + A)| = |Project(A)|`.
/// Implies linearity *I think*.
pub trait AffineGeometricMap<In: GeometricSpace>: LinearGeometricMap<In> {}

/// Marker trait that specifies that a map is linear.
pub trait LinearGeometricMap<In: GeometricSpace>: GeometricMap<In> {
	fn matrix(&self) -> &Mtx3x3;
}

/// Row-major
#[derive(Clone)]
pub struct Mtx3x3(pub [Vct3; 3]);

impl Mtx3x3 {
	pub fn inverse(&self) -> Self {
		let m = |a, b| self[(a, b)];
		let d = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2))
			- m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
			+ m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

		let id = 1.0 / d;

		let mut i = Mtx3x3([Vct3::new(0.0, 0.0, 0.0); 3]);
		let mut minv = |a, b, c| i[(a, b)] = c;
		minv(0, 0, (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * id);
		minv(0, 1, (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * id);
		minv(0, 2, (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * id);
		minv(1, 0, (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * id);
		minv(1, 1, (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * id);
		minv(1, 2, (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * id);
		minv(2, 0, (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * id);
		minv(2, 1, (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * id);
		minv(2, 2, (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * id);
		i
	}
}

impl std::ops::Index<(usize, usize)> for Mtx3x3 {
	type Output = R;

	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		self.0[row].nth(col)
	}
}

impl std::ops::IndexMut<(usize, usize)> for Mtx3x3 {
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		self.0[row].nth_mut(col)
	}
}

impl std::ops::Mul<Mtx3x3> for &Mtx3x3 {
	type Output = Mtx3x3;

	fn mul(self, mut rhs: Mtx3x3) -> Self::Output {
		rhs.set_column_n::<0>(self * rhs.column_n::<0>());
		rhs.set_column_n::<1>(self * rhs.column_n::<1>());
		rhs.set_column_n::<2>(self * rhs.column_n::<2>());
		rhs
	}
}

impl std::ops::Mul<&Mtx3x3> for &Mtx3x3 {
	type Output = Mtx3x3;

	fn mul(self, rhs: &Mtx3x3) -> Self::Output {
		self * rhs.clone()
	}
}

impl std::ops::Mul<Mtx3x3> for Mtx3x3 {
	type Output = Mtx3x3;

	fn mul(self, rhs: Mtx3x3) -> Self::Output {
		&self * rhs
	}
}

impl std::ops::Mul<Vct3> for &Mtx3x3 {
	type Output = Vct3;

	fn mul(self, rhs: Vct3) -> Self::Output {
		Vct3 {
			x: rhs.dot(self.0[0]),
			y: rhs.dot(self.0[1]),
			z: rhs.dot(self.0[2]),
		}
	}
}

impl Mtx3x3 {
	pub const fn column_n<const N: usize>(&self) -> Vct3 {
		Vct3 {
			x: *self.0[0].nth(N),
			y: *self.0[1].nth(N),
			z: *self.0[2].nth(N),
		}
	}

	pub const fn set_column_n<const N: usize>(&mut self, v: Vct3) {
		*self.0[0].nth_mut(N) = v.x;
		*self.0[1].nth_mut(N) = v.y;
		*self.0[2].nth_mut(N) = v.z;
	}

	pub fn rotation_x(a: R) -> Self {
		let (s, c) = a.sin_cos();
		Self([
			Vct3::new(1.0, 0.0, 0.0),
			Vct3::new(0.0, c, -s),
			Vct3::new(0.0, s, c),
		])
	}
	pub fn rotation_y(a: R) -> Self {
		let (s, c) = a.sin_cos();
		Self([
			Vct3::new(c, 0.0, s),
			Vct3::new(0.0, 1.0, 0.0),
			Vct3::new(-s, 0.0, c),
		])
	}
	pub fn rotation_z(a: R) -> Self {
		let (s, c) = a.sin_cos();
		Self([
			Vct3::new(c, -s, 0.0),
			Vct3::new(s, c, 0.0),
			Vct3::new(0.0, 0.0, 1.0),
		])
	}
}

pub struct OrthoProjection {
	mtx: Mtx3x3,
	backdir: Vct3,
}

impl OrthoProjection {
	pub fn new(mtx: Mtx3x3) -> Self {
		let imtx = mtx.inverse();
		let backdir = &imtx * Vct3::new(0.0, 0.0, 1.0);
		Self { mtx, backdir }
	}
}

impl AffineGeometricMap<GSpace3D> for OrthoProjection {}
impl LinearGeometricMap<GSpace3D> for OrthoProjection {
	fn matrix(&self) -> &Mtx3x3 {
		&self.mtx
	}
}
impl GeometricMap<GSpace3D> for OrthoProjection {
	type Out = GSpace2D;

	fn get_back_dir(
		&self,
		_: <GSpace3D as GeometricSpace>::Point,
	) -> <GSpace3D as GeometricSpace>::Vector {
		self.backdir
	}

	fn project_point(&self, p: Pnt3) -> Pnt2 {
		let p = &self.mtx * GSpace3D::radius_vector(p);
		Pnt2 { x: p.x, y: p.y }
	}

	fn project_vector(&self, p: Vct3) -> Vct2 {
		let p = &self.mtx * p;
		Vct2 { x: p.x, y: p.y }
	}

	fn project_line(&self, l: (Pnt3, Vct3)) -> (Pnt2, Vct2) {
		(self.project_point(l.0), self.project_vector(l.1))
	}

	fn project_ray(&self, p: (Pnt3, Vct3)) -> (Pnt2, Vct2) {
		(self.project_point(p.0), self.project_vector(p.1))
	}
}

pub mod tess {
	use crate::*;

	pub trait Intersecting<Other> {
		type Intersection;
		fn intersect(&self, other: &Other) -> Self::Intersection;
	}

	mod util {
		pub struct True;
		pub struct False;
		pub trait Same<U> {
			type Result = False;
		}
		impl<T> Same<T> for T {
			type Result = True;
		}
	}
	impl<T, U> Intersecting<T> for U
	where
		T: Intersecting<U> + util::Same<U, Result = util::False>,
	{
		type Intersection = T::Intersection;
		fn intersect(&self, other: &T) -> Self::Intersection {
			other.intersect(self)
		}
	}

	type Line2 = <GSpace2D as GeometricSpace>::Line;
	type Line3 = <GSpace3D as GeometricSpace>::Line;
	// type Plane3 = <GSpace3D as GeometricSpace>::Plane;

	#[allow(non_camel_case_types)]
	pub enum Line3_Self_Intersection {
		/// The two lines are the same.
		Same(Line3),
		/// A single point of intersection, respective parameters.
		Point(Pnt3, R, R),
		/// No intersection
		None,
	}

	#[allow(non_camel_case_types)]
	pub enum Line2_Self_Intersection {
		/// The lines are identical.
		Same(Line2),
		/// Point, t1, t2.
		Point(Pnt2, R, R),
		/// The lines are parallel.
		None,
	}

	impl Intersecting<Line3> for Line3 {
		type Intersection = Line3_Self_Intersection;

		fn intersect(&self, other: &Line3) -> Self::Intersection {
			if self.0.approx_eq(other.0) && self.1.approx_eq(other.1) {
				Self::Intersection::Same(*self)
			} else {
				todo!()
			}
		}
	}

	impl Intersecting<Line2> for Line2 {
		type Intersection = Line2_Self_Intersection;

		fn intersect(&self, other: &Line2) -> Self::Intersection {
			if self.0.approx_eq(other.0) {
				if self.1.approx_eq(other.1) {
					Self::Intersection::Same(*self)
				} else {
					Self::Intersection::None
				}
			} else {
				// We need to solve this:
				// ⎧ d1x*t1 + o1x = px
				// ⎨ d2x*t2 + o2x = px
				// ⎨ d1y*t1 + o1y = py
				// ⎩ d2y*t2 + o2y = py

				let (c1, d1) = *self;
				let (c2, d2) = *other;
				if (d1.y * d2.x).approx_eq(d1.x * d2.y) || d2.y.approx_eq(0.0) {
					return Self::Intersection::None;
				}
				let t1 = (c1.y * d2.x - c2.y * d2.x + (c2.x - c1.x) * d2.y) / (d1.x * d2.y - d1.y * d2.x);
				let t2 = (c1.y * d1.x - c2.y * d1.x + (c2.x - c1.x) * d1.y) / (d1.x * d2.y - d1.y * d2.x);
				Self::Intersection::Point(c1 + d1 * t1, t1, t2)
			}
		}
	}
}

fn aabb_between(a: Pnt2, b: Pnt2) -> (Pnt2, Pnt2) {
	(
		Pnt2 {
			x: a.x.min(b.x),
			y: a.y.min(b.y),
		},
		Pnt2 {
			x: a.x.max(b.x),
			y: a.y.max(b.y),
		},
	)
}

pub type Tri3 = (Pnt3, Pnt3, Pnt3);

pub fn möller_trumbore_intersection(
	origin: Pnt3,
	direction: Vct3,
	triangle: Tri3,
) -> Option<Pnt3> {
	let e1 = triangle.1 - triangle.0;
	let e2 = triangle.2 - triangle.0;

	let ray_cross_e2 = direction.cross(e2);
	let det = e1.dot(ray_cross_e2);

	if det.approx_eq(0.0) {
		return None; // This ray is parallel to this triangle.
	}

	let inv_det = 1.0 / det;
	let s = origin - triangle.0;
	let u = inv_det * s.dot(ray_cross_e2);
	if u < 0.0 || u > 1.0 {
		return None;
	}

	let s_cross_e1 = s.cross(e1);
	let v = inv_det * direction.dot(s_cross_e1);
	if v < 0.0 || u + v > 1.0 {
		return None;
	}
	// At this stage we can compute t to find out where the intersection point is on the line.
	let t = inv_det * e2.dot(s_cross_e1);

	if t > 1e-6 {
		let intersection_point = origin + direction * t;
		Some(intersection_point)
	} else {
		// This means that there is a line intersection but not a ray intersection.
		None
	}
}

pub mod svg {
	use crate::*;
	use std::borrow::Cow;

	const SVGNS: Option<&str> = Some("http://www.w3.org/2000/svg");

	#[derive(Default, Debug)]
	pub struct Style<'a> {
		pub fill: Option<Cow<'a, str>>,
		pub stroke: Option<Cow<'a, str>>,
		pub stroke_width: Option<f32>,
		pub stroke_dasharray: Option<f32>,
		pub non_scaling_stroke: Option<bool>,
		pub transform_matrix: Option<[R; 6]>,
	}

	impl Style<'_> {
		fn assign_attrs(&self, n: &web_sys::Element) -> Result<(), wasm_bindgen::JsValue> {
			if let Some(v) = &self.fill {
				n.set_attribute("fill", v)?;
			}
			if let Some(v) = &self.stroke {
				n.set_attribute("stroke", v)?;
			}
			if let Some(v) = self.stroke_width {
				n.set_attribute("stroke-width", &v.to_string())?;
			}
			if let Some(v) = self.stroke_dasharray {
				n.set_attribute("stroke-dasharray", &v.to_string())?;
			}
			if let Some(v) = self.non_scaling_stroke {
				n.set_attribute("vector-effect", if v {
					"non-scaling-stroke"
				} else {
					"none"
				})?;
			} else {
				n.set_attribute("vector-effect", "non-scaling-stroke")?;
			}
			if let Some(v) = &self.transform_matrix {
				n.set_attribute("transform", &format!(
					"matrix({} {} {} {} {} {})",
					v[0], v[3], v[1], v[4], v[2], v[5]
				))?;
			}
			Ok(())
		}
	}

	pub fn root(doc: &web_sys::Document, w: f32) -> web_sys::Element {
		let doc = doc.create_element_ns(SVGNS, "svg").unwrap();
		doc
			.set_attribute("viewBox", &format!("{} {} {} {}", -w, -w, 2.0 * w, 2.0 * w))
			.unwrap();
		doc.set_attribute("width", "300").unwrap();
		doc.set_attribute("height", "300").unwrap();
		doc.set_attribute("fill", "black").unwrap();
		// doc.set_attribute("vector-effect", "non-scaling-stroke").unwrap();
		// doc.set_attribute("xmlns", SVGNS.unwrap()).unwrap();
		doc
	}

	pub fn rect(doc: &web_sys::Document, o: Pnt2, d: Vct2, st: &Style) -> web_sys::Element {
		let n = doc.create_element_ns(SVGNS, "rect").unwrap();
		n.set_attribute("x", &o.x.to_string()).unwrap();
		n.set_attribute("y", &o.y.to_string()).unwrap();
		n.set_attribute("width", &d.x.to_string()).unwrap();
		n.set_attribute("height", &d.y.to_string()).unwrap();
		st.assign_attrs(&n).unwrap();
		n
	}

	pub fn line(doc: &web_sys::Document, a: Pnt2, b: Pnt2, st: &Style) -> web_sys::Element {
		let n = doc.create_element_ns(SVGNS, "line").unwrap();
		n.set_attribute("x1", &a.x.to_string()).unwrap();
		n.set_attribute("y1", &a.y.to_string()).unwrap();
		n.set_attribute("x2", &b.x.to_string()).unwrap();
		n.set_attribute("y2", &b.y.to_string()).unwrap();
		st.assign_attrs(&n).unwrap();
		n
	}

	pub fn circle(doc: &web_sys::Document, c: Pnt2, r: R, st: &Style) -> web_sys::Element {
		let n = doc.create_element_ns(SVGNS, "circle").unwrap();
		n.set_attribute("cx", &c.x.to_string()).unwrap();
		n.set_attribute("cy", &c.y.to_string()).unwrap();
		n.set_attribute("r", &r.to_string()).unwrap();
		st.assign_attrs(&n).unwrap();
		n
	}

	pub fn triangle(
		doc: &web_sys::Document,
		a: Pnt2,
		b: Pnt2,
		c: Pnt2,
		st: &Style,
	) -> web_sys::Element {
		let n = doc.create_element_ns(SVGNS, "path").unwrap();
		n.set_attribute(
			"d",
			&format!("M{},{} L{},{} L{},{} z", a.x, a.y, b.x, b.y, c.x, c.y),
		)
		.unwrap();
		st.assign_attrs(&n).unwrap();
		n
	}

	pub fn polygon(
		doc: &web_sys::Document,
		mut it: impl Iterator<Item = Pnt2>,
		st: &Style,
	) -> web_sys::Element {
		let n = doc.create_element_ns(SVGNS, "path").unwrap();
		let first = it
			.next()
			.expect("Expected at least one point in svg::polygon");
		let mut res = format!("M{},{}", first.x, first.y);
		for p in it {
			res += &format!(" L{},{}", p.x, p.y);
		}
		res += " z";
		n.set_attribute("d", &res).unwrap();
		st.assign_attrs(&n).unwrap();
		n
	}
}

pub mod render {
	use crate::*;
	use std::{borrow::Cow, collections::BTreeSet};

	pub enum PrimitiveStyle {
		Major,
		Minor,
		AxisX,
		AxisY,
		AxisZ,
	}

	impl PrimitiveStyle {
		fn is_transparent(&self) -> bool {
			match self {
				PrimitiveStyle::Major => false,
				PrimitiveStyle::Minor => false,
				PrimitiveStyle::AxisX => true,
				PrimitiveStyle::AxisY => true,
				PrimitiveStyle::AxisZ => true,
			}
		}
	}

	pub enum PrimitiveShape<'a> {
		/// A single point.
		Point(Pnt3),
		/// A line segment between two points.
		Segment(Pnt3, Pnt3),
		/// A 2D circle: `(center, normal)`, `radius = normal.magnitude2()`.
		Circle(Pnt3, Vct3),
		/// A 2D polygon.
		Polygon(Cow<'a, [Pnt3]>),
		/// A sphere.
		Sphere(Pnt3, R),
	}

	pub trait Shape<In: GeometricSpace> {
		fn render<'a>(&self) -> impl Iterator<Item = (PrimitiveStyle, PrimitiveShape<'a>)>;
	}

	pub struct SphereShape {
		pub center: Pnt3,
		pub radius: R,
	}

	impl SphereShape {
		pub fn new(center: Pnt3, radius: R) -> Self {
			Self {
				center,
				radius,
			}
		}
	}

	impl Shape<GSpace3D> for SphereShape {
		fn render<'a>(&self) -> impl Iterator<Item = (PrimitiveStyle, PrimitiveShape<'a>)> {
			std::iter::once((
				PrimitiveStyle::Major,
				PrimitiveShape::Sphere(self.center, self.radius),
			))
		}
	}

	pub struct PolyhedronShape<'a> {
		pub vertices: Cow<'a, [Pnt3]>,
		pub edges: Cow<'a, [(usize, usize)]>,
		pub faces: Cow<'a, [Cow<'a, [usize]>]>,
	}

	impl PolyhedronShape<'_> {
		pub fn octahedron(center: Pnt3, r: R) -> Self {
			let off = GSpace3D::radius_vector(center);
			Self {
				vertices: vec![
					Pnt3::new(-1.0, -1.0, 0.0).onto_sphere(r) + off,
					Pnt3::new(1.0, -1.0, 0.0).onto_sphere(r) + off,
					Pnt3::new(1.0, 1.0, 0.0).onto_sphere(r) + off,
					Pnt3::new(-1.0, 1.0, 0.0).onto_sphere(r) + off,
					Pnt3::new(0.0, 0.0, 1.0).onto_sphere(r) + off,
					Pnt3::new(0.0, 0.0, -1.0).onto_sphere(r) + off,
				]
				.into(),
				edges: (vec![
					(0, 1),
					(1, 2),
					(2, 3),
					(3, 0),
					(0, 4),
					(1, 4),
					(2, 4),
					(3, 4),
					(0, 5),
					(1, 5),
					(2, 5),
					(3, 5),
				])
				.into(),
				faces: (vec![
					(&[0, 1, 4]).into(),
					(&[1, 2, 4]).into(),
					(&[2, 3, 4]).into(),
					(&[3, 0, 4]).into(),
					(&[0, 1, 5]).into(),
					(&[1, 2, 5]).into(),
					(&[2, 3, 5]).into(),
					(&[3, 0, 5]).into(),
				])
				.into(),
			}
		}
	}

	impl Shape<GSpace3D> for PolyhedronShape<'_> {
		fn render<'a>(&self) -> impl Iterator<Item = (PrimitiveStyle, PrimitiveShape<'a>)> {
			std::iter::from_coroutine(
				#[coroutine]
				|| {
					for (ai, bi) in self.edges.iter() {
						let (a, b) = (self.vertices[*ai], self.vertices[*bi]);
						yield (PrimitiveStyle::Major, PrimitiveShape::Segment(a, b))
					}
					for idxs in self.faces.iter() {
						yield (
							PrimitiveStyle::Major,
							PrimitiveShape::Polygon(
								idxs
									.iter()
									.map(|i| self.vertices[*i])
									.collect::<Vec<_>>()
									.into(),
							),
						)
					}
				},
			)
		}
	}

	enum TesselatedPrimitiveShape {
		Segment(Pnt3, BTreeSet<isize>, Pnt3),
		Sphere(Pnt3, R, Vec<(Vct3, R)>),
	}

	pub struct SvgRenderer<'a, 'm, M: GeometricMap<GSpace3D, Out = GSpace2D>> {
		doc: &'a web_sys::Document,
		root: web_sys::Element,
		map: &'m M,
		prims: Vec<(PrimitiveStyle, PrimitiveShape<'m>)>,
	}

	impl<
		'a,
		'm,
		M: AffineGeometricMap<GSpace3D, Out = GSpace2D>
	> SvgRenderer<'a, 'm, M> {
		pub fn new(doc: &'a web_sys::Document, map: &'m M) -> Self {
			let root = svg::root(doc, 2.0);
			Self {
				doc,
				root,
				map,
				prims: vec![],
			}
		}

		pub fn get_root(self) -> web_sys::Element {
			self.root
		}

		fn style_of(&self, s: &PrimitiveStyle) -> svg::Style {
			match s {
				PrimitiveStyle::Major => svg::Style {
					stroke_width: Some(3.0),
					..Default::default()
				},
				PrimitiveStyle::Minor => svg::Style {
					stroke_width: Some(2.0),
					..Default::default()
				},
				PrimitiveStyle::AxisX => svg::Style {
					stroke: Some("red".into()),
					stroke_width: Some(2.0),
					..Default::default()
				},
				PrimitiveStyle::AxisY => svg::Style {
					stroke: Some("green".into()),
					stroke_width: Some(2.0),
					..Default::default()
				},
				PrimitiveStyle::AxisZ => svg::Style {
					stroke: Some("blue".into()),
					stroke_width: Some(2.0),
					..Default::default()
				},
			}
		}

		// fn project_primitive(&self, p: PrimitiveShape) ->  {
		// 	match p {
		// 		PrimitiveShape::Point(st, p) => {
		// 			let st =self.style_of(st);
		// 			svg::circle(self.doc, p, st.stroke_width.unwrap(), &st)
		// 		},
		// 		PrimitiveShape::Segment(st, a, b) => svg::line(self.doc, a, b, &self.style_of(st)),
		// 		PrimitiveShape::Circle(st, c, r) => svg::circle(self.doc, c, r, &self.style_of(st)),
		// 		PrimitiveShape::Polygon(st, ps) => svg::polygon(self.doc, ps.iter().copied(), &self.style_of(st)),
		// 	}
		// }

		pub fn render_axes(&mut self, r: R) {
			self.prims.push((
				PrimitiveStyle::AxisX,
				PrimitiveShape::Segment(Pnt3::zero(), Pnt3::new(r, 0.0, 0.0)),
			));
			self.prims.push((
				PrimitiveStyle::AxisY,
				PrimitiveShape::Segment(Pnt3::zero(), Pnt3::new(0.0, r, 0.0)),
			));
			self.prims.push((
				PrimitiveStyle::AxisZ,
				PrimitiveShape::Segment(Pnt3::zero(), Pnt3::new(0.0, 0.0, r)),
			));
		}

		pub fn render_shape(&mut self, s: &impl Shape<GSpace3D>) {
			self.prims.extend(s.render());
			// for prim in  {
			// 	self.root.append_child(&self.make_primitive(prim)).unwrap();
			// }
		}

		fn tesselate(
			&self
		) -> impl Iterator<Item = (&PrimitiveStyle, TesselatedPrimitiveShape)> + Clone {
			// let segment_count = self.prims.iter()
			// 	.filter(|x| matches!(x.1, PrimitiveShape::Segment(..))).count();
			let mut tess_lines = self
				.prims
				.iter()
				.filter_map(|x| match x.1 {
					PrimitiveShape::Segment(a, b) => Some((&x.0, a, b)),
					_ => None,
				})
				.map(|(st, a, b)| {
					let mut ts = std::collections::BTreeSet::new();
					ts.insert(1024);
					(st, (a, ts, b))
				})
				.collect::<Vec<_>>();

			for i in 0..tess_lines.len() {
				for j in 0..tess_lines.len() {
					if i == j {
						continue;
					}
					let [(st1, (a1, ts, b1)), (st2, (a2, _, b2))] = tess_lines.get_disjoint_mut([i, j]).unwrap();
					if st1.is_transparent() || st2.is_transparent() {
						continue;
					}
					let pa1 = self.map.project_point(*a1);
					let pb1 = self.map.project_point(*b1);
					let pa2 = self.map.project_point(*a2);
					let pb2 = self.map.project_point(*b2);
					let (bba1, bbb1) = aabb_between(pa1, pb1);
					let (bba2, bbb2) = aabb_between(pa2, pb2);
					let pl1 = GSpace2D::line_between(pa1, pb1);
					let pl2 = GSpace2D::line_between(pa2, pb2);
					use tess::Intersecting;
					match pl1.intersect(&pl2) {
						tess::Line2_Self_Intersection::Point(p, t1, _) => {
							if p.is_inside(bba1, bbb1) && p.is_inside(bba2, bbb2) {
								ts.insert((t1 * 1024.0) as isize);
							}
						}
						_ => {}
					}
				}
			}

			tess_lines
				.into_iter()
				.map(|(st, (a, ts, b))| (st, TesselatedPrimitiveShape::Segment(a, ts, b)))
				.chain(self.prims.iter().filter_map(|(st, pr)| match pr {
					PrimitiveShape::Sphere(center, radius) =>
						Some((st, TesselatedPrimitiveShape::Sphere(*center, *radius, vec![]))),
					_ => None,
				}))
		}

		pub fn draw(&self) {
			let tesselated = self.tesselate();
			let tess_lines = tesselated.clone().filter_map(|(st, pr)| match pr {
				TesselatedPrimitiveShape::Segment(a, tess, c) => Some((st, a, tess, c)),
				_ => None
			});
			for (_, (st, a, tess, c)) in tess_lines.enumerate() {
				let mut o = a;
				for bi in tess.iter() {
					let t = *bi as f32 / 1024.0;
					let b = a + (c - a) * t;
					if o.approx_eq(b) {
						continue;
					}
					let m = o + (b - o) * 0.5;
					let x = self
						.prims
						.iter()
						.filter_map(|(_, x)| match x {
							PrimitiveShape::Polygon(pts) => Some(pts),
							_ => None,
						})
						.any(|pts| {
							// let pts = (fa, fb, fc)
							möller_trumbore_intersection(m, self.map.get_back_dir(m), (pts[0], pts[1], pts[2]))
								.is_some()
						});
					let po = self.map.project_point(o);
					let pb = self.map.project_point(b);
					self
						.root
						.append_child(&svg::line(
							self.doc,
							po,
							pb,
							&svg::Style {
								stroke_dasharray: x.then_some(5.0),
								..self.style_of(st)
							},
						))
						.unwrap();
					o = b;
				}
			}

			let tess_spheres = tesselated.clone().filter_map(|(st, pr)| match pr {
				TesselatedPrimitiveShape::Sphere(c, r, tess) => Some((st, c, r, tess)),
				_ => None
			});

			for (_, (st, c, r, tess)) in tess_spheres.enumerate() {
				self.root.append_child(
					&svg::circle(
						self.doc,
						self.map.project_point(c),
						// TODO
						r, // self.map.project_vector(Vct3::new(r, 0.0, 0.0)).magnitude(),
						&svg::Style {
							fill: Some("none".into()),
							..self.style_of(st)
						}
					)
				).unwrap();
				for (norm, dist) in tess {
					let mtx = self.map.matrix();

					// t.
				}
			}

			for pts in self.prims.iter().filter_map(|(_, x)| match x {
				PrimitiveShape::Polygon(pts) => Some(pts),
				_ => None,
			}) {
				self
					.root
					.append_child(&svg::polygon(
						self.doc,
						pts.iter().map(|p| self.map.project_point(*p)),
						&svg::Style {
							stroke: Some("none".into()),
							fill: Some("#668bb623".into()),
							..Default::default()
						},
					))
					.unwrap();
			}
		}
	}
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn init() {
	std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn example_1(rot_x: f32, rot_y: f32, rot_z: f32) -> wasm_bindgen::JsValue {
	let proj = OrthoProjection::new(
		Mtx3x3::rotation_x(rot_x) * Mtx3x3::rotation_y(rot_y) * Mtx3x3::rotation_z(rot_z),
	);

	let web_window = web_sys::window().unwrap();
	let web_doc = web_window.document().unwrap();
	let mut svgr = render::SvgRenderer::new(&web_doc, &proj);
	svgr.render_axes(2.0);
	svgr.render_shape(&render::PolyhedronShape::octahedron(Pnt3::zero(), 1.0));
	svgr.render_shape(&render::SphereShape::new(Pnt3::zero(), 1.0));
	svgr.draw();
	svgr.get_root().into()
}
