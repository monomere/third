import setupWasm, { example_1, init } from "./pkg/third.js";

await setupWasm();

init();

/** @type {SVGSVGElement} */
let svg = document.body.appendChild(example_1(65.0, 0.0, 25.0));


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
	const rX = Number.parseFloat(inpX.value) / 180 * Math.PI;
	const rY = Number.parseFloat(inpY.value) / 180 * Math.PI;
	const rZ = Number.parseFloat(inpZ.value) / 180 * Math.PI;
	const newSvg = example_1(rX, rY, rZ);
	svg.replaceWith(newSvg);
	svg = newSvg;
	requestAnimationFrame(rerender);
});

inpX.oninput = () => hadInput = true;
inpY.oninput = () => hadInput = true;
inpZ.oninput = () => hadInput = true;
