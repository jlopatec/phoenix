import React, { Suspense, useMemo } from "react";
import { Outlet } from "react-router";
import { css } from "@emotion/react";

import { Flex, Icon, Icons } from "@arizeai/components";

import { Loading } from "@phoenix/components";
import {
  Brand,
  DocsLink,
  GitHubLink,
  NavBreadcrumb,
  NavLink,
  SideNavbar,
  ThemeToggle,
  TopNavbar,
} from "@phoenix/components/nav";
import { useFunctionality } from "@phoenix/contexts/FunctionalityContext";

const layoutCSS = css`
  display: flex;
  direction: row;
  height: 100vh;
  overflow: hidden;
`;

const mainViewCSS = css`
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  height: 100%;
  overflow: hidden;
  padding-left: var(--px-nav-collapsed-width);
`;
const contentCSS = css`
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
`;

const bottomLinksCSS = css`
  display: flex;
  flex-direction: column;
  margin: 0;
  list-style: none;
  gap: var(--ac-global-dimension-size-50);
  padding-inline-start: 0;
`;

const sideLinksCSS = css`
  display: flex;
  flex-direction: column;
  gap: var(--ac-global-dimension-size-50);
`;

export function Layout() {
  return (
    <div css={layoutCSS} data-testid="layout">
      <SideNav />
      <div css={mainViewCSS}>
        <TopNavbar>
          <NavBreadcrumb />
        </TopNavbar>
        <div data-testid="content" css={contentCSS}>
          <Suspense fallback={<Loading />}>
            <Outlet />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

function SideNav() {
  const hasInferences = useMemo(() => {
    return window.Config.hasInferences;
  }, []);
  const { authenticationEnabled } = useFunctionality();
  return (
    <SideNavbar>
      <Brand />
      <Flex direction="column" justifyContent="space-between" flex="1 1 auto">
        <ul css={sideLinksCSS}>
          {hasInferences && (
            <li>
              <NavLink
                to="/model"
                text="Model"
                icon={<Icon svg={<Icons.CubeOutline />} />}
              />
            </li>
          )}
          <li>
            <NavLink
              to="/projects"
              text="Projects"
              icon={<Icon svg={<Icons.GridOutline />} />}
            />
          </li>
          <li>
            <NavLink
              to="/datasets"
              text="Datasets"
              icon={<Icon svg={<Icons.DatabaseOutline />} />}
            />
          </li>
          <li>
            <NavLink
              to="/apis"
              text="APIs"
              icon={<Icon svg={<Icons.Code />} />}
            />
          </li>
        </ul>
        <ul css={bottomLinksCSS}>
          <li>
            <NavLink
              to="/settings"
              text="Settings"
              icon={<Icon svg={<Icons.SettingsOutline />} />}
            />
          </li>
          <li>
            <DocsLink />
          </li>
          <li>
            <GitHubLink />
          </li>
          <li>
            <ThemeToggle />
          </li>
          {authenticationEnabled && (
            <li>
              <NavLink
                to="/profile"
                text="Profile"
                icon={<Icon svg={<Icons.PersonOutline />} />}
              />
            </li>
          )}
        </ul>
      </Flex>
    </SideNavbar>
  );
}
