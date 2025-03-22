import setupWasm, { example_1, init } from "./pkg/third.js";

await setupWasm();

init();

const toRadians = a => a / 180.0 * Math.PI;

/** @type {SVGSVGElement} */
let svg = document.body.appendChild(
	example_1(toRadians(65.0), 0.0, toRadians(25.0)));


function createAxis(name, initialValue) {
	const div = document.body.appendChild(document.createElement("div"));
	div.classList.add("slider");
	div.append(`${name}: -180°`);
	const inp = div.appendChild(document.createElement("input"));
	div.append("+180°");
	inp.type = "range";
	inp.min = -180;
	inp.max = 180;
	inp.value = initialValue;
	return inp
}

const inpX = createAxis("X", 65.0);
const inpY = createAxis("Y", 0.0);
const inpZ = createAxis("Z", 25.0);

let hadInput = false;

let rerender;
requestAnimationFrame(rerender = () => {
	if (!hadInput) { requestAnimationFrame(rerender); return; }
	hadInput = false;
	const rX = toRadians(Number.parseFloat(inpX.value));
	const rY = toRadians(Number.parseFloat(inpY.value));
	const rZ = toRadians(Number.parseFloat(inpZ.value));
	const newSvg = example_1(rX, rY, rZ);
	svg.replaceWith(newSvg);
	svg = newSvg;
	requestAnimationFrame(rerender);
});

inpX.oninput = () => hadInput = true;
inpY.oninput = () => hadInput = true;
inpZ.oninput = () => hadInput = true;
